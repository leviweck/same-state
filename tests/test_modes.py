"""Retrieval mode switch: dense / fused / fused+rerank. Config sets it (default fused); the CLI flag overrides."""
import json
import pathlib

import pytest
import yaml

from plm_kit import cli, config, rerank
from plm_kit.pipeline import Retriever, resolve_mode
from tests.fakes import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_data" / "docs"


def rcfg(**over):
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["retrieval"]
    c.update(over)
    return c


def test_default_mode_is_fused():
    assert rcfg()["mode"] == "fused"
    assert resolve_mode(rcfg()) == ("fused", None)


@pytest.mark.parametrize("mode", ["dense", "fused", "fused+rerank"])
def test_config_mode_and_flag_override(mode):
    assert resolve_mode(rcfg(mode=mode))[0] == mode
    other = "dense" if mode != "dense" else "fused"
    assert resolve_mode(rcfg(mode=mode), other)[0] == other  # flag beats config


def test_bad_mode_rejected(tmp_path):
    with pytest.raises(ValueError):
        resolve_mode(rcfg(mode="bm25"))  # bm25 is an eval arm, not a user mode
    with pytest.raises(ValueError):
        resolve_mode(rcfg(), "hybrid")
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    c["retrieval"]["mode"] = "rerank"
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        config.load_config(f)
    assert cli.main(["--config", str(f), "health"]) == 2


@pytest.mark.parametrize("over", [{"top_n": 0}, {"weight": 0}])
def test_rerank_mode_falls_back_when_switched_off(over, monkeypatch):
    c = rcfg(mode="fused+rerank")
    c["reranker"] = dict(c["reranker"], **over)
    mode, note = resolve_mode(c)
    assert mode == "fused" and note


def test_kill_switch_beats_rerank_mode(monkeypatch):
    monkeypatch.setenv("PLM_KIT_RERANKER", "off")
    assert resolve_mode(rcfg(mode="fused+rerank"))[0] == "fused"
    assert resolve_mode(rcfg(), "fused+rerank")[0] == "fused"


def _ret(mode, calls):
    def scorer(q, docs):
        calls.append(len(docs))
        return [1.0 if "Caro-Kann" in d else 0.0 for d in docs]
    return Retriever(DOCS, rcfg(mode=mode), embedder=HashEmbedder(), scorer=scorer)


def test_search_follows_config_mode():
    calls = []
    assert _ret("fused", calls).search("chess openings for black").chunks is not None
    assert calls == []                                   # fused: reranker never called
    _ret("dense", calls).search("chess openings for black")
    assert calls == []
    res = _ret("fused+rerank", calls).search("chess openings for black")
    assert calls and res.docs[0] == "chess-openings.md"  # fused+rerank: scorer used


def test_modes_differ_in_what_they_run():
    r = Retriever(DOCS, rcfg(), embedder=HashEmbedder())
    dense = r.search("factory reset AX-7731", mode="dense")
    fused = r.search("factory reset AX-7731", mode="fused")
    assert dense.timings_ms["bm25"] < 1.0 and dense.best_dense is not None
    assert fused.docs[0] == "home-wifi.md"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(rerank, "get_reranker", lambda c: pytest.fail("reranker loaded outside fused+rerank"))
    return tmp_path / "cache"


def test_cli_flag_overrides_config_mode(cache, capsys, tmp_path, monkeypatch):
    seen = []
    real = cli.build_retriever

    def spy(cfg, docs=None, cache=None, out=None, mode=None, **kw):
        ret, info = real(cfg, docs, cache, out, mode, **kw)
        seen.append(info["mode"])
        return ret, info

    monkeypatch.setattr(cli, "build_retriever", spy)
    assert cli.main(["ask", "factory reset AX-7731"]) == 0
    assert seen[-1] == "fused"
    # dense needs the embedding model; without it the CLI says so instead of silently using BM25
    assert cli.main(["ask", "factory reset AX-7731", "--mode", "dense"]) == 2
    assert "plm-kit download" in capsys.readouterr().err
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    c["retrieval"]["mode"] = "dense"
    f = tmp_path / "dense.yaml"
    f.write_text(yaml.safe_dump(c))
    assert cli.main(["--config", str(f), "ask", "factory reset AX-7731"]) == 2      # config says dense
    assert cli.main(["--config", str(f), "ask", "factory reset AX-7731", "--mode", "fused"]) == 0  # flag wins
    assert seen[-1] == "fused"


def test_cli_rerank_mode_without_export_falls_back(cache, capsys, monkeypatch):
    monkeypatch.setattr(rerank, "get_reranker", lambda c: None)  # not exported -> None (fail closed)
    assert cli.main(["ask", "factory reset AX-7731", "--mode", "fused+rerank"]) == 0
    err = capsys.readouterr().err
    assert "using fused" in err


def test_cli_rejects_unknown_mode(cache):
    with pytest.raises(SystemExit):
        cli.main(["ask", "x", "--mode", "bm25"])


def test_eval_per_mode_option(monkeypatch, capsys):
    monkeypatch.delenv("PLM_KIT_EMBED_ONNX", raising=False)
    monkeypatch.delenv("PLM_KIT_RERANK_ONNX", raising=False)
    from eval import run_eval

    rep = run_eval.main(modes=["bm25", "dense"])
    assert rep["modes"] == ["bm25"] and "dense" in rep["skipped_modes"]
    with pytest.raises(ValueError):
        run_eval.main(modes=["nope"])
    capsys.readouterr()
    assert cli.main(["eval", "--modes", "bm25", "--no-grid"]) == 0
    assert json.loads(capsys.readouterr().out)["modes"] == ["bm25"]
    assert cli.main(["eval", "--modes", "bm25,warp"]) == 2


def test_eval_per_mode_with_fake_models(monkeypatch):
    from eval import run_eval

    def fake_build(cfg):
        sc = run_eval.CachingScorer(lambda q, d: [float(len(x) % 7) for x in d])
        return Retriever(DOCS, cfg, embedder=HashEmbedder(), scorer=sc), {"model_load_s": 0.0}

    monkeypatch.setattr(run_eval, "build", fake_build)
    rep = run_eval.main(modes=["dense", "fused+rerank"], grid=False)
    assert rep["modes"] == ["dense", "fused+rerank"] and rep["skipped_modes"] == {}
    assert set(rep["full"]) == {"dense", "fused+rerank"} and "tuned_on_calib" in rep["threshold"]
