"""Synthetic eval: recall@1/3/5, hit@1/3/5, MRR, not-in-notes caught / false-gated, CPU latency.

Arms: bm25 | dense | fused | fused+rerank over a grid (top_n x weight). Dense/rerank arms need locally exported
ONNX models (from `plm-kit download`):
  PLM_KIT_EMBED_ONNX=<cache>/granite-embed-r2-onnx   PLM_KIT_RERANK_ONNX=<cache>/granite-r2-onnx (optional)
Without them only the bm25 arm runs. No network is used here.

The not-in-notes threshold is tuned on a deterministic, stratified 70% calibration split and reported on the
held-out 30% test split. Retrieval metrics are reported on the full set and on the test split.
Per-mode: --modes bm25,dense,fused,fused+rerank picks the arms (default: all that can run). fused+rerank uses
the configured top_n/weight; the grid (on unless --no-grid) adds the top_n x weight cells.
--fixed-threshold skips the split tuning and gates every query at the configured not_in_notes_threshold.
Usage: python -m eval.run_eval [--json out.json] [--modes ...] [--no-grid] [--no-tune] [--fixed-threshold]"""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import statistics
import time

import yaml

from plm_kit.metrics import hit_at_k, mrr, recall_at_k
from plm_kit.pipeline import Retriever

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_data" / "docs"
GOLD = ROOT / "eval" / "golden_synthetic.jsonl"
KS = (1, 3, 5)
K_MAX = max(KS)
GRID_TOP_N = (6, 10, 20)
GRID_WEIGHT = (0.25, 0.5, 1.0)
ARMS = ("bm25", "dense", "fused", "fused+rerank")


def load_cfg():
    return yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["retrieval"]


def load_gold(path=None):
    return [json.loads(x) for x in pathlib.Path(path or GOLD).read_text().splitlines() if x.strip()]


def split(rows, frac=0.7):
    """Deterministic stratified split by sha256(id): returns (calib_ids, test_ids)."""
    calib, test = set(), set()
    for flag in (True, False):
        grp = sorted((r for r in rows if r["answerable"] is flag), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        cut = round(len(grp) * frac)
        calib |= {r["id"] for r in grp[:cut]}
        test |= {r["id"] for r in grp[cut:]}
    return calib, test


class CachingScorer:
    """Wraps the real cross-encoder; caches (query, text) scores so weight variants reuse them."""

    def __init__(self, fn):
        self.fn, self.cache, self.live = fn, {}, True

    def __call__(self, q, docs):
        miss = [d for d in docs if (q, d) not in self.cache]
        if miss and self.live:
            for d, s in zip(miss, self.fn(q, miss)):
                self.cache[(q, d)] = s
        return [self.cache[(q, d)] for d in docs]


def run_arm(ret, rows, mode, k=K_MAX):
    """One pass: per-query ranked docs, best dense cosine, in_notes, latency."""
    out = {}
    for r in rows:
        t0 = time.perf_counter()
        res = ret.search(r["query"], mode=mode, top_k=k)
        out[r["id"]] = {"docs": res.docs, "best": res.best_dense, "in_notes": res.in_notes,
                        "ms": (time.perf_counter() - t0) * 1e3}
    return out


def metrics(runs, rows, ids, gate=None):
    """gate: callable(run) -> in_notes. Default uses the run's own in_notes."""
    sel = [r for r in rows if r["id"] in ids]
    ans = [r for r in sel if r["answerable"]]
    nin = [r for r in sel if not r["answerable"]]
    m = {}
    for k in KS:
        m[f"recall@{k}"] = round(sum(recall_at_k(runs[r["id"]]["docs"], r["relevant"], k) for r in ans) / len(ans), 3)
        m[f"hit@{k}"] = round(sum(hit_at_k(runs[r["id"]]["docs"], r["relevant"], k) for r in ans) / len(ans), 3)
    m["mrr"] = round(sum(mrr(runs[r["id"]]["docs"], r["relevant"]) for r in ans) / len(ans), 3)
    g = gate or (lambda x: x["in_notes"])
    m["nin_caught"] = f"{sum(not g(runs[r['id']]) for r in nin)}/{len(nin)}"
    m["false_gated"] = f"{sum(not g(runs[r['id']]) for r in ans)}/{len(ans)}"
    lat = sorted(runs[r["id"]]["ms"] for r in sel)
    m["lat_mean_ms"] = round(statistics.mean(lat), 2)
    m["lat_p95_ms"] = round(lat[int(0.95 * (len(lat) - 1))], 2)
    return m


def tune_threshold(runs, rows, ids):
    """Maximize balanced accuracy (mean of: answerable kept, unanswerable caught) on the calibration ids.
    Plain accuracy is useless here: with ~85% answerable, "never gate" already scores ~0.85.
    Ties -> midpoint of the first contiguous best range."""
    pts = [(runs[r["id"]]["best"], r["answerable"]) for r in rows if r["id"] in ids]
    n_a = sum(a for _, a in pts)
    n_n = len(pts) - n_a
    cands = sorted({round(b, 4) for b, _ in pts} | {round(b + 1e-4, 4) for b, _ in pts})

    def bal(t):
        keep = sum(b >= t for b, a in pts if a) / n_a
        caught = sum(b < t for b, a in pts if not a) / max(n_n, 1)
        return (keep + caught) / 2

    acc = [(round(bal(t), 6), t) for t in cands]
    best = max(a for a, _ in acc)
    i = next(n for n, (a, _) in enumerate(acc) if a == best)
    j = i
    while j + 1 < len(acc) and acc[j + 1][0] == best:
        j += 1
    return round((acc[i][1] + acc[j][1]) / 2, 4), round(best, 3)


def build(cfg, docs=None):
    embedder = scorer = None
    info = {"model_load_s": 0.0}
    e_dir, r_dir = os.environ.get("PLM_KIT_EMBED_ONNX"), os.environ.get("PLM_KIT_RERANK_ONNX")
    if e_dir:
        from plm_kit.dense import FastembedEmbedder
        from plm_kit.models import resolve_embedder

        t0 = time.perf_counter()
        embedder = FastembedEmbedder(resolve_embedder(cfg["dense"])[1], e_dir)
        info["model_load_s"] = round(time.perf_counter() - t0, 2)
    if r_dir:
        from plm_kit import rerank

        fn = rerank.get_reranker(dict(cfg["reranker"], enabled=True, onnx_dir=r_dir))
        scorer = CachingScorer(fn) if fn else None
    from plm_kit.store import note_mtimes

    docs = pathlib.Path(docs or DOCS)
    t0 = time.perf_counter()
    ret = Retriever(docs, cfg, embedder=embedder, scorer=scorer, mtimes=note_mtimes(docs))
    info.update(index_build_s=round(time.perf_counter() - t0, 2), chunks=len(ret.chunks),
                docs=len({c.doc for c in ret.chunks}), newest_copy=ret.newest.stats)
    return ret, info


def main(k=5, json_out=None, cfg=None, rows=None, grid=True, tune=True, modes=None, docs=None, gold=None,
         fixed_threshold=False):
    cfg = cfg or load_cfg()
    rows = rows or load_gold(gold)
    modes = list(modes or ARMS)
    if set(modes) - set(ARMS):
        raise ValueError(f"unknown eval modes {sorted(set(modes) - set(ARMS))}; choose from {list(ARMS)}")
    calib, test = split(rows)
    all_ids = {r["id"] for r in rows}
    ret, info = build(cfg, docs) if docs else build(cfg)
    need = {"bm25": True, "dense": ret.dense is not None, "fused": ret.dense is not None,
            "fused+rerank": ret.dense is not None and ret.scorer is not None}
    runs, skipped = {}, {}
    for m in modes:
        if not need[m]:
            skipped[m] = "needs PLM_KIT_EMBED_ONNX" + (" and PLM_KIT_RERANK_ONNX" if m == "fused+rerank" else "")
            continue
        if m == "fused+rerank":
            ret.scorer("warm-up", ["warm-up"])
            ret.scorer.cache.clear()
        runs[m] = run_arm(ret, rows, m)
    if ret.dense and ret.scorer and grid and "fused+rerank" in modes:
        base = copy.deepcopy(cfg)
        ret.scorer("warm-up", ["warm-up"])  # first ONNX call pays session setup; keep it out of latency
        for n in GRID_TOP_N:  # live pass per top_n measures real CE latency; other weights reuse cached scores
            ret.scorer.cache.clear()  # each top_n's live pass scores all of its pairs (no reuse from a smaller top_n)
            for w in GRID_WEIGHT:
                ret.cfg = copy.deepcopy(base)
                ret.cfg["reranker"].update(top_n=n, weight=w)
                live = run_arm(ret, rows, "fused+rerank")
                if w == GRID_WEIGHT[0]:
                    lat = {i: v["ms"] for i, v in live.items()}
                for i in live:
                    live[i]["ms"] = lat[i]
                runs[f"rerank n{n} w{w}"] = live
        ret.cfg = base
    report = {"queries": len(rows), "answerable": sum(r["answerable"] for r in rows),
              "not_in_notes": sum(not r["answerable"] for r in rows),
              "split": {"calib": len(calib), "test": len(test)}, "info": info, "modes": list(runs),
              "skipped_modes": skipped,
              "config": {"rrf_k": cfg["fusion"]["k"], "stale": cfg.get("stale"), "per_note_cap": cfg.get("per_note_cap"),
                         "preset": cfg.get("preset"), "dedupe": cfg.get("dedupe"),
                         "not_in_notes_gate": cfg.get("not_in_notes_gate", "block"), "reranker": {k: cfg["reranker"].get(k) for k in ("top_n", "weight")},
                         "configured_threshold": cfg.get("not_in_notes_threshold")}}
    gate = None
    gate_arm = next((a for a in ("dense", "fused", "fused+rerank") if a in runs), None)
    if gate_arm and fixed_threshold:
        thr = cfg.get("not_in_notes_threshold")
        if thr is None:
            raise ValueError("--fixed-threshold needs retrieval.not_in_notes_threshold")
        report["threshold"] = {"fixed": thr}

        def gate(x, t=thr):
            return x["best"] >= t
    elif gate_arm and tune:
        thr, cal_acc = tune_threshold(runs[gate_arm], rows, calib)
        report["threshold"] = {"tuned_on_calib": thr, "calib_balanced_accuracy": cal_acc}

        def gate(x, t=thr):
            return x["best"] >= t
    report["full"] = {a: metrics(r, rows, all_ids, None if a == "bm25" else gate) for a, r in runs.items()}
    report["test"] = {a: metrics(r, rows, test, None if a == "bm25" else gate) for a, r in runs.items()}
    if gate and fixed_threshold:
        report["threshold"]["full_gate"] = {k: report["full"][gate_arm][k] for k in ("nin_caught", "false_gated")}
    elif gate:
        g = runs[gate_arm]
        report["threshold"]["test_gate"] = {
            "nin_caught": report["test"][gate_arm]["nin_caught"], "false_gated": report["test"][gate_arm]["false_gated"],
            "configured_threshold_on_test": metrics(g, rows, test, lambda x: x["best"] >= cfg["not_in_notes_threshold"])
            if cfg.get("not_in_notes_threshold") is not None else None}
    print(json.dumps(report, indent=1))
    if json_out:
        pathlib.Path(json_out).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--fixed-threshold", action="store_true", help="gate at the configured threshold, no tuning")
    ap.add_argument("--modes", help="comma list: " + ",".join(ARMS))
    ap.add_argument("--newest-copy", choices=["off", "demote", "drop"])
    ap.add_argument("--docs")
    ap.add_argument("--gold")
    a = ap.parse_args()
    c = load_cfg()
    if a.newest_copy:
        c["dedupe"] = dict(c.get("dedupe") or {}, newest_copy=a.newest_copy)
    main(a.k, a.json, cfg=c, grid=not a.no_grid, tune=not a.no_tune, docs=a.docs, gold=a.gold,
         fixed_threshold=a.fixed_threshold, modes=[m.strip() for m in a.modes.split(",") if m.strip()] if a.modes else None)
