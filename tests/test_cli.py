"""CLI tests. No network: the Hugging Face fetch and the ONNX export are mocked."""
import builtins
import io
import json
import pathlib

import pytest

from plm_kit import cli, download, models, paths, safety, store
from tests.fakes import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return cli.load_config()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    c = tmp_path / "cache"
    monkeypatch.setenv("PLM_KIT_CACHE", str(c))
    return c


@pytest.fixture
def no_hub(monkeypatch):
    real = builtins.__import__

    def guard(name, *a, **kw):
        if name.split(".")[0] in ("huggingface_hub", "fastembed", "optimum", "transformers"):
            raise AssertionError(f"unexpected import of {name}")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guard)


def mock_net(monkeypatch):
    calls = []
    monkeypatch.setattr(download, "fetch", lambda spec, dest: calls.append(("fetch", spec["hf_id"], spec["revision"])) or dest)
    monkeypatch.setattr(download, "export_onnx", lambda src, task, out: calls.append(("export", task, str(out))))
    return calls


def test_download_prints_notice_before_fetch_and_uses_pinned_revisions(cfg, cache, monkeypatch):
    calls = mock_net(monkeypatch)
    buf = io.StringIO()
    done = download.run(cfg, ["embed", "reranker"], assume_yes=True, out=buf)
    text = buf.getvalue()
    assert text.index("LICENSE NOTICE") < text.index("ok: embed")
    assert download.THIRD_PARTY in text and "apache-2.0" in text
    assert ("fetch", "ibm-granite/granite-embedding-english-r2", models.EMBEDDERS["granite-embed-r2"]["revision"]) in calls
    assert ("fetch", "ibm-granite/granite-embedding-reranker-english-r2", models.RERANKERS["granite-r2"]["revision"]) in calls
    assert all(pathlib.Path(p).is_relative_to(cache) for p in done.values())


def test_download_requires_explicit_yes(cfg, cache, monkeypatch):
    calls = mock_net(monkeypatch)
    buf = io.StringIO()
    assert download.run(cfg, ["embed"], out=buf, inp=lambda _: "y") == {}
    assert calls == [] and "Nothing was downloaded" in buf.getvalue()


def test_download_eof_or_ctrl_c_cancels_cleanly(cfg, cache, monkeypatch):
    calls = mock_net(monkeypatch)
    for exc in (EOFError, KeyboardInterrupt):
        def inp(_prompt, exc=exc):
            raise exc
        buf = io.StringIO()
        assert download.run(cfg, ["embed", "gguf"], out=buf, inp=inp) == {}
        assert calls == [] and "Cancelled. Nothing was downloaded." in buf.getvalue()


def test_generator_only_when_requested(cfg, cache, monkeypatch):
    calls = mock_net(monkeypatch)
    download.run(cfg, ["embed"], assume_yes=True, out=io.StringIO())
    assert not any("Qwen" in c[1] for c in calls if c[0] == "fetch")
    download.run(cfg, ["generator"], assume_yes=True, out=io.StringIO())
    assert ("fetch", "Qwen/Qwen3-8B", models.GENERATORS["qwen3-8b"]["revision"]) in calls
    assert not any(c[0] == "export" and "qwen" in c[2] for c in calls)


def test_cli_download_command_mocked(cache, monkeypatch, capsys):
    mock_net(monkeypatch)
    assert cli.main(["download", "--yes"]) == 0
    assert "LICENSE NOTICE" in capsys.readouterr().out


def test_cache_inside_repo_refused(monkeypatch):
    monkeypatch.setenv("PLM_KIT_CACHE", str(ROOT / "models"))
    with pytest.raises(ValueError):
        paths.cache_dir()
    assert cli.main(["health"]) == 2


def test_custom_embedder_via_config_and_validation():
    spec = {"name": "my-embed", "hf_id": "org/model", "revision": "a" * 40, "pooling": "mean", "dim": 384,
            "normalize": True, "license": "mit", "license_url": "https://opensource.org/license/mit"}
    name, s = models.resolve_embedder({"model": "granite-embed-r2", "custom": spec})
    assert name == "my-embed" and s["hf_id"] == "org/model"
    with pytest.raises(ValueError):
        models.resolve_embedder({"custom": dict(spec, revision="main")})
    with pytest.raises(ValueError):
        models.resolve_embedder({"custom": dict(spec, license="cc-by-nc-4.0")})


def test_index_and_ask_bm25_only_without_models(cache, no_hub, capsys):
    assert cli.main(["index"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["dense"] is False and info["rebuilt"] is True and pathlib.Path(info["index_dir"]).is_relative_to(cache)
    assert cli.main(["index"]) == 0
    assert json.loads(capsys.readouterr().out)["rebuilt"] is False
    assert cli.main(["ask", "factory reset AX-7731", "--k", "2"]) == 0
    assert "home-wifi.md" in capsys.readouterr().out.split("\n")[0]
    assert cli.main(["ask", "zzqx qqzz"]) == 0
    assert "Not in your notes" in capsys.readouterr().out


def test_store_rebuilds_when_docs_or_embedder_change(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\nalpha beta\n")
    idx = tmp_path / "idx"
    spec = {"hf_id": "x/y", "revision": "b" * 40}
    assert store.build_or_load(docs, idx, HashEmbedder(), spec)[2] is True
    assert store.build_or_load(docs, idx, HashEmbedder(), spec)[2] is False
    assert store.build_or_load(docs, idx, HashEmbedder(), dict(spec, revision="c" * 40))[2] is True
    (docs / "b.md").write_text("# B\ngamma\n")
    chunks, matrix, rebuilt = store.build_or_load(docs, idx, HashEmbedder(), dict(spec, revision="c" * 40))
    assert rebuilt and len(chunks) == 2 and matrix.shape[0] == 2


def test_ask_answer_without_generator_fails_cleanly(cache, no_hub, capsys):
    # the opt-in CUDA path (the default is auto; see tests/test_local_llm.py for its fallback)
    assert cli.main(["ask", "starter feeding ratio", "--answer", "--backend", "transformers_peft"]) == 2
    assert "download --include-generator" in capsys.readouterr().err


def test_health(cache, capsys):
    assert cli.main(["health"]) == 0
    h = json.loads(capsys.readouterr().out)
    assert h["fusion_k"] == 60 and h["reranker"]["used"] is False
    assert h["retrieval_mode"] == {"configured": "fused", "effective": "fused", "note": None} and h["embedder"]["ready"] is False


def test_draft_check(tmp_path, capsys):
    f = tmp_path / "d.txt"
    f.write_text("Reach me at someone@example.com, card 4111 1111 1111 1111. This is CONFIDENTIAL.")
    assert cli.main(["draft-check", str(f)]) == 1
    out = capsys.readouterr().out
    assert "email" in out and "card_like" in out and "confidential" in out
    f.write_text("Happy to share the garden watering schedule next week.")
    assert cli.main(["draft-check", str(f)]) == 0
    assert safety.check("order 1234 5678 9012 3456") == []  # fails Luhn -> not flagged


def test_eval_command_bm25_only(monkeypatch, capsys):
    monkeypatch.delenv("PLM_KIT_EMBED_ONNX", raising=False)
    assert cli.main(["eval", "--no-grid"]) == 0
    assert '"bm25"' in capsys.readouterr().out
