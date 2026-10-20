"""Real-model eval. Needs locally exported ONNX (scripts/export_onnx.py); skipped otherwise (CI has no models).
Run: PLM_KIT_EMBED_ONNX=... [PLM_KIT_RERANK_ONNX=...] pytest -m model"""
import os

import pytest

pytestmark = [pytest.mark.model,
              pytest.mark.skipif(not os.environ.get("PLM_KIT_EMBED_ONNX"), reason="PLM_KIT_EMBED_ONNX not set")]


def test_real_dense_and_fused_quality():
    pytest.importorskip("fastembed")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from eval.run_eval import main

    rep = main(grid=bool(os.environ.get("PLM_KIT_RERANK_ONNX")))
    full, test = rep["full"], rep["test"]
    assert full["dense"]["hit@5"] >= 0.9 and full["fused"]["hit@5"] >= 0.9
    assert full["fused"]["recall@5"] >= full["bm25"]["recall@5"]
    assert test["dense"]["mrr"] >= 0.85
    assert "tuned_on_calib" in rep["threshold"]
    if os.environ.get("PLM_KIT_RERANK_ONNX"):
        assert len([a for a in full if a.startswith("rerank")]) == 9
