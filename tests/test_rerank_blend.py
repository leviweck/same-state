import os

from plm_kit import rerank
from plm_kit.fusion import rrf

CFG = {"enabled": True, "model": "granite-r2", "top_n": 3, "weight": 5.0, "k": 1}
TEXTS = {0: "alpha", 1: "beta live", 2: "beta archive copy"}


def test_fake_scorer_lifts_relevant_and_keeps_archive_below():
    order = rerank.blend("beta", [0, 1, 2], TEXTS, CFG, scorer=lambda q, d: [0.0, 1.0, 2.0],
                         archived={2}, archive_weight=0.5)
    assert order[0] == 1 and order.index(1) < order.index(2)


def test_scorer_error_returns_order_unchanged():
    assert rerank.blend("beta", [0, 1, 2], TEXTS, CFG, scorer=lambda q, d: 1 / 0) == [0, 1, 2]


def test_wrong_length_scores_unchanged():
    assert rerank.blend("beta", [0, 1, 2], TEXTS, CFG, scorer=lambda q, d: [1.0]) == [0, 1, 2]


def test_off_by_default_and_env_kill_switch():
    assert rerank.get_reranker({"enabled": False, "top_n": 20, "weight": 1.0}) is None
    os.environ["PLM_KIT_RERANKER"] = "off"
    try:
        assert not rerank.enabled(CFG)
    finally:
        del os.environ["PLM_KIT_RERANKER"]


def test_only_allowlisted_models():
    assert all(v["license"] in ("apache-2.0", "mit") for v in rerank.RERANKERS.values())
    assert not any("marco" in v["hf_id"].lower() for v in rerank.RERANKERS.values())


def test_rrf_basic():
    assert rrf([["a", "b"], ["b", "c"]], k=60)[0] == "b"
