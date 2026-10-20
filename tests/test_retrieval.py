import pathlib

import pytest
import yaml

from plm_kit.bm25 import BM25Index, fts_query
from plm_kit.docs import load_chunks
from plm_kit.pipeline import Retriever, rrf_scores
from tests.fakes import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_data" / "docs"


def cfg(**over):
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["retrieval"]
    c.update(over)
    return c


def test_chunks_load_and_archive_flag():
    ch = load_chunks(DOCS)
    assert len(ch) >= 25
    arch = {c.doc for c in ch if c.archived}
    assert arch == {"archive/home-wifi-old.md", "archive/sourdough-old.md"}


@pytest.mark.parametrize("q", ['"; DROP TABLE chunks; --', "NEAR(a b) OR *", 'x" OR "y', "AND", "^col:term"])
def test_fts_query_is_injection_safe(q):
    idx = BM25Index(load_chunks(DOCS))
    idx.search(q)  # must not raise an FTS5 syntax error
    assert all(t.startswith('"') and t.endswith('"') for t in fts_query(q).split(" OR ") if t)


def test_bm25_exact_identifier():
    ch = load_chunks(DOCS)
    top = BM25Index(ch).search("factory reset AX-7731")[0]
    assert ch[top].doc == "home-wifi.md"


def test_rrf_k60_math():
    s = rrf_scores([[1, 2], [2, 3]], k=60)
    assert s[2] == pytest.approx(1 / 62 + 1 / 61) and s[1] == pytest.approx(1 / 61)
    assert cfg()["fusion"]["k"] == 60


def test_archive_copy_ranks_below_live_copy():
    r = Retriever(DOCS, cfg(), embedder=HashEmbedder()).search("guest network LanternGuest", mode="fused")
    assert r.docs.index("home-wifi.md") < r.docs.index("archive/home-wifi-old.md")


def test_gate_uses_best_dense_and_falls_back_to_bm25():
    hi = Retriever(DOCS, cfg(not_in_notes_threshold=1.01), embedder=HashEmbedder())
    assert hi.search("oil change hatchback").in_notes is False
    lo = Retriever(DOCS, cfg(not_in_notes_threshold=0.0), embedder=HashEmbedder())
    assert lo.search("oil change hatchback").in_notes is True
    bm_only = Retriever(DOCS, cfg(), embedder=None)
    assert bm_only.search("oil change hatchback").in_notes is True
    assert bm_only.search("zzqx unrelatedtoken").in_notes is False


def test_default_mode_never_calls_scorer_when_reranker_off():
    calls = []
    r = Retriever(DOCS, cfg(), embedder=HashEmbedder(), scorer=lambda q, d: calls.append(1) or [0.0] * len(d))
    r.search("starter feeding ratio")
    assert calls == []


def test_explicit_rerank_mode_uses_blend():
    def scorer(q, docs):
        return [1.0 if "Caro-Kann" in d else 0.0 for d in docs]

    r = Retriever(DOCS, cfg(), embedder=HashEmbedder(), scorer=scorer)
    res = r.search("chess openings for black", mode="fused+rerank")
    assert res.docs[0] == "chess-openings.md"


def test_eval_runs_bm25_only_without_models(monkeypatch):
    monkeypatch.delenv("PLM_KIT_EMBED_ONNX", raising=False)
    monkeypatch.delenv("PLM_KIT_RERANK_ONNX", raising=False)
    from eval.run_eval import load_gold, main, split

    rep = main()
    assert list(rep["full"]) == ["bm25"]
    a = rep["full"]["bm25"]
    assert {"recall@1", "recall@3", "recall@5", "hit@1", "hit@3", "hit@5", "mrr", "nin_caught", "false_gated"} <= set(a)
    assert a["hit@5"] >= 0.7
    rows = load_gold()
    cal, test = split(rows)
    assert not cal & test and len(cal | test) == len(rows)
    assert 50 <= len(rows) <= 100 and 0.10 <= sum(not r["answerable"] for r in rows) / len(rows) <= 0.20
