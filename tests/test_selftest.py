"""Starter test kit: `plm-kit selftest` on the bundled synthetic notes (no network, no models)."""
import io
import json
import pathlib

from plm_kit import cli, selftest
from tests.fakes import HashEmbedder


def cfg():
    return cli.load_config()


def test_sample_kit_shape():
    notes = sorted(p.relative_to(selftest.NOTES).as_posix() for p in selftest.NOTES.rglob("*.md"))
    assert 20 <= len(notes) <= 40
    assert any(n.startswith(("archive/", "old/")) for n in notes)              # stale copies
    assert any("export" in n or "chat-summar" in n for n in notes)             # derived copies
    rows = selftest.load_gold()
    for r in rows:
        assert all(d in notes for d in r["relevant"] + r.get("stale", [])), r["id"]
    nin = [r for r in rows if not r["answerable"]]
    assert len(nin) >= 5 and len(rows) - len(nin) >= 25
    assert sum(1 for r in rows if r.get("stale")) >= 5
    from plm_kit import voice_lite

    assert len(voice_lite.load_examples(selftest.STYLE_EXAMPLES)) == 6     # the bundled synthetic file is valid


def test_selftest_keyword_only_passes_without_models(tmp_path, monkeypatch):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    buf = io.StringIO()
    out = tmp_path / "card.json"
    assert selftest.run(cfg(), json_out=str(out), out=buf) == 0
    text = buf.getvalue()
    assert "KEYWORD search only" in text and "PASS: selftest" in text
    rep = json.loads(out.read_text())
    assert rep["arm"] == "keyword_only" and rep["notes"] == 33 and rep["pass"] is True
    assert {c["name"] for c in rep["checks"]} == {"recall@3", "mrr", "stale_demoted"}


def test_selftest_fused_with_fake_embedder_scores_the_gate(tmp_path):
    buf = io.StringIO()
    out = tmp_path / "card.json"
    selftest.run(cfg(), json_out=str(out), out=buf, embedder=HashEmbedder())
    rep = json.loads(out.read_text())
    assert rep["arm"] == "fused" and "nin_caught" in rep["metrics"] and "false_gated" in rep["metrics"]


def test_selftest_fails_when_demotion_is_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    c = cfg()
    c["retrieval"]["stale"] = dict(c["retrieval"]["stale"], archive_weight=1.0, derived_weight=1.0)
    buf = io.StringIO()
    assert selftest.run(c, out=buf) == 1
    assert "stale_demoted" in buf.getvalue() and "FAIL" in buf.getvalue()


def test_selftest_answers_with_mock_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    ex = [x for x in selftest.STYLE_EXAMPLES.read_text().splitlines() if x and not x.startswith("#")]
    calls = []

    def chat(messages, max_tokens):   # first voice-lite draft copies an example; the guard must catch it
        calls.append(messages)
        return ex[0] if len(calls) == 2 else "The rate is $65 per hour, the deposit is 50%, backups run at 2 am."

    buf = io.StringIO()
    out = tmp_path / "card.json"
    c = cfg()
    c["generation"]["backend"] = "ollama"
    assert selftest.run(c, answer=True, out=buf, json_out=str(out), chat=chat) == 0
    rep = json.loads(out.read_text())
    rows = rep["answers"]["rows"]
    assert len(rows) == 3 and all(r["voice_lite"]["copies_in_output"] == 0 for r in rows)
    assert rows[0]["voice_lite"]["guard"]["action"] == "regenerated"


def test_selftest_answer_error_is_a_fail_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    c = cfg()
    c["generation"]["backend"] = "llamacpp_server"
    c["generation"]["llamacpp_server"] = {"url": "http://127.0.0.1:9"}
    c["generation"]["local"] = dict(c["generation"]["local"], timeout_s=2)
    buf = io.StringIO()
    assert selftest.run(c, answer=True, out=buf) == 1
    assert "can't reach" in buf.getvalue()


def test_selftest_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "c"))
    assert cli.main(["selftest"]) == 0
    assert "recall@3" in capsys.readouterr().out
    assert not list(pathlib.Path(tmp_path / "c").glob("**/manifest.json"))   # no index written to the user cache
