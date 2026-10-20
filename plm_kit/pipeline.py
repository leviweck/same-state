"""Hybrid retrieval: BM25 (FTS5) + dense, fused with RRF (k=60), optional reranker blend, not-in-notes gate."""
import time
from dataclasses import dataclass, field

from plm_kit import rerank
from plm_kit.bm25 import BM25Index
from plm_kit.config import DEFAULT_MODE, MODES
from plm_kit.dedupe import NewestCopy
from plm_kit.dense import DenseIndex
from plm_kit.docs import load_chunks
from plm_kit.stale import chunk_multipliers


@dataclass
class Result:
    docs: list                 # doc ids, best first (deduplicated from chunks)
    chunks: list               # chunk ids, best first
    in_notes: bool             # False -> answer "not in your notes" (only when the gate is "block")
    best_dense: float | None = None
    timings_ms: dict = field(default_factory=dict)
    low_confidence: bool = False  # best dense match below the threshold (gate "warn" or "block")


def rrf_scores(rankings, k=60):
    score = {}
    for ranked in rankings:
        for r, cid in enumerate(ranked, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
    return score


def resolve_mode(cfg, override=None):
    """Effective retrieval mode and an optional note. `override` (CLI flag) beats retrieval.mode (config).
    fused+rerank drops to fused when PLM_KIT_RERANKER=off or reranker.top_n / weight is 0."""
    mode = override or cfg.get("mode") or DEFAULT_MODE
    if mode not in MODES:
        raise ValueError(f"retrieval mode must be one of {list(MODES)}, got {mode!r}")
    if mode == "fused+rerank" and not rerank.enabled(dict(cfg.get("reranker", {}), enabled=True)):
        return "fused", "reranker switched off (PLM_KIT_RERANKER=off, or top_n/weight is 0); using fused"
    return mode, None


GATE_MODES = ("block", "warn", "off")


def gate_mode(cfg):
    """retrieval.not_in_notes_gate: block (answer "Not in your notes"), warn (show results with a low-confidence
    warning) or off. A null threshold also turns the gate off."""
    g = cfg.get("not_in_notes_gate", "block")
    if g in (False, None):
        g = "off"
    if g not in GATE_MODES:
        raise ValueError(f"retrieval.not_in_notes_gate must be one of {list(GATE_MODES)}")
    return g


def cap_per_note(order, doc_of, cap):
    """Keep at most `cap` chunks per note, preserving order."""
    seen, out = {}, []
    for c in order:
        n = seen.get(doc_of[c], 0)
        if n < cap:
            out.append(c)
            seen[doc_of[c]] = n + 1
    return out


class Retriever:
    def __init__(self, docs_dir, cfg, embedder=None, scorer=None, chunks=None, matrix=None, mtimes=None):
        """docs_dir is read only when `chunks` isn't given (a persisted index passes chunks + matrix)."""
        self.cfg = cfg
        self.chunks = chunks if chunks is not None else load_chunks(docs_dir)
        self.text = {c.cid: c.text for c in self.chunks}
        self.doc = {c.cid: c.doc for c in self.chunks}
        self.archived = frozenset(c.cid for c in self.chunks if c.archived)
        self.mult = chunk_multipliers(self.chunks, cfg)   # stale-copy demotion, see plm_kit/stale.py
        self.newest = NewestCopy(self.chunks, cfg, mtimes)  # optional newest-copy step (off by default)
        self.bm25 = BM25Index(self.chunks) if cfg.get("bm25", {}).get("enabled", True) else None
        dense_on = cfg.get("dense", {}).get("enabled", True) and embedder is not None
        self.dense = DenseIndex(self.chunks, embedder, matrix=matrix) if dense_on else None
        self.scorer = scorer  # (query, texts) -> scores; None -> rerank.get_reranker(cfg) decides

    def search(self, query, mode=None, top_k=None):
        """mode: 'dense' | 'fused' | 'fused+rerank' ('bm25' for eval). None -> retrieval.mode from config
        (default 'fused'), resolved by resolve_mode(). An explicit mode is used as given."""
        t, cfg = {}, self.cfg
        if mode is None:
            mode = resolve_mode(cfg)[0]
        if mode not in (*MODES, "bm25"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        top_k = top_k or cfg.get("top_k", 10)
        k = cfg.get("fusion", {}).get("k", 60)
        cap = cfg.get("per_note_cap", 2)
        floor = cfg.get("dense_floor")
        mult = self.mult
        bm, dn, sims, allsims = [], [], [], None
        t0 = time.perf_counter()
        if self.bm25 and mode in ("bm25", "fused", "fused+rerank"):
            bm = self.bm25.search(query)
        t["bm25"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        if self.dense and mode in ("dense", "fused", "fused+rerank"):
            dn, sims, allsims = self.dense.search(query, return_all=True)
        t["dense"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        if mode == "bm25":
            order = bm
        elif mode == "dense":
            order = dn
        else:
            s = rrf_scores([r for r in (bm, dn) if r], k=k)
            s = {c: v * mult.get(c, 1.0) for c, v in s.items()}   # demotion AFTER fusion; nothing dropped
            s = self.newest.apply(s, self.doc)                     # newest-copy step (no-op unless enabled)
            order = sorted(s, key=lambda c: (-s[c], c))
        if floor is not None and allsims is not None:          # optional dense-similarity floor (default off)
            order = [c for c in order if float(allsims[c]) >= floor]
        if cap:
            order = cap_per_note(order, self.doc, cap)
        t["fuse"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        if mode == "fused+rerank":
            rcfg = dict(cfg.get("reranker", {}), enabled=True)
            scorer = self.scorer or rerank.get_reranker(rcfg)
            order = rerank.blend(query, order, self.text, rcfg, scorer, multipliers=mult)
        t["rerank"] = (time.perf_counter() - t0) * 1e3

        best = sims[0] if sims else None
        thr = cfg.get("not_in_notes_threshold")
        gate = gate_mode(cfg)
        low = best is not None and thr is not None and best < thr
        if gate == "block" and best is not None and thr is not None:
            in_notes = not low              # gate on the best dense match, before any reranking
        else:
            in_notes = bool(order)          # gate off/warn, or BM25-only: any hit counts
        docs = []
        for c in order:
            if self.doc[c] not in docs:
                docs.append(self.doc[c])
        return Result(docs[:top_k], order[:top_k], in_notes, best, t, low_confidence=low and gate != "off")
