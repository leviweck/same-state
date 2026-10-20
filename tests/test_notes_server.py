"""Read-only notes server (plm_kit/notes_server.py): the tool list is exactly the six read-only tools, nothing can
send/write/reach the network, and every tool works against the synthetic eval notes (sample_data/docs)."""
import asyncio
import json
import os
import pathlib
import re
import sys

import pytest

from plm_kit import config, notes_server
from plm_kit.pipeline import Retriever

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_data" / "docs"
STYLE = ROOT / "starter_kit" / "style_examples.txt"
SEND_LIKE = re.compile(r"send|post|publish|submit|write|delete|remove|update|create|upload|email|mail|reply|"
                       r"forward|share|exec|shell|run|fetch|http|request|save|move|rename|schedule|invite|pay",
                       re.IGNORECASE)

mcp = pytest.importorskip("mcp")


def _cfg(**mcp_overrides):
    cfg = config.load_config()
    cfg = dict(cfg, generation=dict(cfg["generation"], voice_lite=dict(cfg["generation"]["voice_lite"],
                                                                         examples_file=str(STYLE))))
    cfg["integrations"] = {"mcp": dict(cfg.get("integrations", {}).get("mcp") or {}, **mcp_overrides)}
    return cfg


def _service(**kw):
    cfg = _cfg(**kw)
    return notes_server.NotesService(cfg, lambda: (Retriever(str(DOCS), cfg["retrieval"]),
                                                   {"mode": "fused", "dense": False}))


def _call(srv, name, args=None):
    r = asyncio.run(srv.call_tool(name, args or {}))
    assert not r.is_error, r
    return json.loads(r.content[0].text)


@pytest.fixture(scope="module")
def srv():
    return notes_server.build_server(_service())


def test_tool_list_is_exactly_the_read_only_set(srv):
    tools = asyncio.run(srv.list_tools())
    assert sorted(t.name for t in tools) == sorted(notes_server.READ_ONLY_TOOLS)
    for t in tools:
        a = t.annotations
        assert a.read_only_hint is True and a.destructive_hint is False and a.open_world_hint is False, t.name


def test_no_send_write_or_network_tools(srv):
    """Adding a tool whose name sounds like an action (send, post, write, delete, fetch, ...) fails here."""
    names = [t.name for t in asyncio.run(srv.list_tools())]
    assert not [n for n in names if SEND_LIKE.search(n)], names
    src = (ROOT / "plm_kit" / "notes_server.py").read_text()
    for banned in ("urllib", "requests", "socket", "http.client", "smtplib", "subprocess", "open(", "write_text"):
        assert banned not in src, f"notes_server.py must stay read-only and offline: found {banned!r}"


def test_search_notes_finds_the_synthetic_note(srv):
    out = _call(srv, "search_notes", {"query": "factory reset AX-7731", "k": 2})
    assert out["in_notes"] and out["results"][0]["note"] == "home-wifi.md"
    assert "15 seconds" in out["results"][0]["excerpt"] and len(out["results"]) <= 2


def test_search_notes_clips_k_and_excerpts():
    svc = _service(max_k=1, max_excerpt_chars=20)
    out = svc.search_notes("router reset", k=50)
    assert len(out["results"]) == 1 and out["results"][0]["excerpt"].endswith("[...]")
    with pytest.raises(ValueError):
        svc.search_notes("   ")


def test_get_note_reads_only_indexed_notes(srv):
    out = _call(srv, "get_note", {"note": "home-wifi.md"})
    assert out["note"] == "home-wifi.md" and "AX-7731" in out["text"]
    from mcp.server.mcpserver.exceptions import ToolError

    for bad in ("../README.md", "/etc/passwd", str(DOCS / "home-wifi.md"), "nope.md"):
        with pytest.raises(ToolError, match="unknown note id"):    # sent to the agent as a tool error
            asyncio.run(srv.call_tool("get_note", {"note": bad}))


def test_get_note_is_bounded():
    out = _service(max_note_chars=30).get_note("home-wifi.md")
    assert out["truncated"] and len(out["text"]) <= 36


def test_draft_check_flags_personal_data(srv):
    out = _call(srv, "draft_check", {"text": "Call me at 555-123-4567 or mail a.person@example.com"})
    kinds = {f["kind"] for f in out["findings"]}
    assert out["flagged"] and {"phone", "email"} <= kinds
    assert not _call(srv, "draft_check", {"text": "See you Tuesday."})["flagged"]


def test_copy_check_reports_positions_only(srv):
    line = next(ln for ln in STYLE.read_text().splitlines() if len(ln.split()) >= 12 and not ln.startswith("#"))
    out = _call(srv, "copy_check", {"text": "Intro. " + line + " Outro."})
    assert out["checked"] and out["flagged"] and set(out["flags"][0]) == {"example", "words", "start", "end"}
    assert line not in json.dumps(out)
    assert not _call(srv, "copy_check", {"text": "Something completely different."})["flagged"]


def test_voice_examples_off_by_default_and_opt_in(srv):
    assert _call(srv, "voice_examples") == {"enabled": False, "examples": [],
                                             "reason": "integrations.mcp.expose_voice_examples is false (the default)"}
    on = _service(expose_voice_examples=True).voice_examples()
    assert on["enabled"] and 5 <= len(on["examples"]) <= 8


def test_status(srv):
    out = _call(srv, "status")
    assert out["read_only"] is True and out["tools"] == list(notes_server.READ_ONLY_TOOLS)
    assert out["notes"] >= 5 and out["voice_examples_exposed"] is False


def test_settings_validation():
    with pytest.raises(ValueError):
        notes_server.settings({"integrations": {"mcp": {"max_k": 0}}})


def test_stdio_end_to_end(tmp_path):
    """Start `plm-kit mcp` as a real subprocess and talk MCP over stdio (BM25-only; no models, no network)."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = dict(os.environ, PLM_KIT_CACHE=str(tmp_path / "cache"), PYTHONPATH=str(ROOT))
    params = StdioServerParameters(command=sys.executable, args=["-m", "plm_kit.cli", "mcp", "--docs", str(DOCS)],
                                   env=env, cwd=str(ROOT))

    async def go():
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = sorted(t.name for t in (await s.list_tools()).tools)
            res = await s.call_tool("search_notes", {"query": "factory reset AX-7731", "k": 1})
            return names, json.loads(res.content[0].text)

    names, out = asyncio.run(asyncio.wait_for(go(), 120))
    assert names == sorted(notes_server.READ_ONLY_TOOLS)
    assert out["results"][0]["note"] == "home-wifi.md"


def test_mcp_command_without_the_sdk_gives_install_message(monkeypatch, capsys):
    """mcp is optional (requirements-mcp.txt): without it `plm-kit mcp` exits 2 with a clear install line."""
    import importlib.util
    import sys

    from plm_kit import cli, notes_server

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: None if n.split(".")[0] == "mcp" else real(n, *a))
    for m in [m for m in sys.modules if m == "mcp" or m.startswith("mcp.")]:
        monkeypatch.delitem(sys.modules, m)
    assert notes_server.mcp_available() is False
    assert cli.main(["mcp"]) == 2
    err = capsys.readouterr().err
    assert "pip install -r requirements-mcp.txt" in err and "Traceback" not in err


def test_mcp_is_an_optional_requirement():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    core = (root / "requirements.txt").read_text() + (root / "requirements.lock").read_text()
    assert not any(ln.split("==")[0].strip().lower() in ("mcp", "mcp-types", "pydantic", "starlette", "uvicorn")
                   for ln in core.splitlines() if "==" in ln and not ln.lstrip().startswith("#"))
    extra = (root / "requirements-mcp.txt").read_text()
    assert "mcp==2.3.0" in extra
    allow = root / "release-allowlist.txt"   # maintainer file; not in a release download
    if allow.is_file():
        assert "requirements-mcp.txt" in allow.read_text()
