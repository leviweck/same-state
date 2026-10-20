"""A disabled reranker must never import fastembed, download, or load a model."""
import builtins
import os

import pytest

from plm_kit import rerank

DISABLED = [
    {"enabled": False, "model": "granite-r2", "top_n": 20, "weight": 1.0, "onnx_dir": "/nonexistent"},
    {"enabled": True, "model": "granite-r2", "top_n": 0, "weight": 1.0, "onnx_dir": "/nonexistent"},
    {"enabled": True, "model": "granite-r2", "top_n": 20, "weight": 0.0, "onnx_dir": "/nonexistent"},
]


@pytest.fixture
def tripwires(monkeypatch):
    calls = []
    real_import = builtins.__import__

    def guarded_import(name, *a, **kw):
        if name.split(".")[0] in ("fastembed", "huggingface_hub", "onnxruntime", "transformers", "optimum"):
            calls.append(("import", name))
            raise AssertionError(f"disabled reranker imported {name}")
        return real_import(name, *a, **kw)

    def no_load(*a, **kw):
        calls.append(("load", a))
        raise AssertionError("disabled reranker tried to load a model")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(rerank, "_load", no_load)
    monkeypatch.setattr(rerank.os.path, "isfile", lambda p: calls.append(("stat", p)) or False)
    monkeypatch.setitem(rerank._STATE, "tried", False)
    monkeypatch.setitem(rerank._STATE, "fn", None)
    return calls


@pytest.mark.parametrize("cfg", DISABLED)
def test_disabled_never_imports_downloads_or_loads(tripwires, cfg):
    assert rerank.get_reranker(cfg) is None
    assert tripwires == []
    assert rerank._STATE["tried"] is False


def test_env_kill_switch_beats_enabled_config(tripwires, monkeypatch):
    monkeypatch.setenv("PLM_KIT_RERANKER", "off")
    assert rerank.get_reranker({"enabled": True, "model": "granite-r2", "top_n": 20, "weight": 1.0}) is None
    assert tripwires == []


def test_enabled_without_export_fails_closed_without_loading(tripwires):
    cfg = {"enabled": True, "model": "granite-r2", "top_n": 20, "weight": 1.0, "onnx_dir": "/nonexistent"}
    assert rerank.get_reranker(cfg) is None
    assert rerank._STATE["error"] == "FileNotFoundError"
    assert not any(c[0] in ("import", "load") for c in tripwires)


def test_default_config_is_off():
    yaml = pytest.importorskip("yaml")
    path = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")
    with open(path) as f:
        r = yaml.safe_load(f)["retrieval"]
    assert r["mode"] == "fused" and "enabled" not in r["reranker"]  # reranker runs only in mode fused+rerank
    assert r["fusion"]["k"] == 60
