"""Cursor + Grok Bot integration files exist, parse, and keep the read-only / never-send rules."""
import json
import pathlib
import re

from plm_kit import notes_server

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_cursor_mcp_config_launches_the_notes_server():
    cfg = json.loads((ROOT / ".cursor" / "mcp.json").read_text())
    srv = cfg["mcpServers"]["plm-kit-notes"]
    assert srv["type"] == "stdio" and srv["args"] == ["-m", "plm_kit.cli", "mcp"]
    assert "url" not in srv                                      # stdio only: no port, no remote URL


def test_cursor_rules_have_front_matter():
    rules = sorted((ROOT / ".cursor" / "rules").glob("*"))
    assert rules and all(p.suffix == ".mdc" for p in rules)     # .md files are ignored by Cursor
    for p in rules:
        text = p.read_text()
        assert text.startswith("---\n") and "alwaysApply:" in text.split("---", 2)[1], p.name
    assert "alwaysApply: true" in (ROOT / ".cursor" / "rules" / "safety-gates.mdc").read_text()


def test_agents_md_lists_the_gates():
    text = (ROOT / "AGENTS.md").read_text()
    for gate in ("pytest", "ruff", "gitleaks", "check_licenses.py", "test_content_rules.py"):
        assert gate in text


def test_skill_never_sends_and_stops_for_the_owner():
    text = (ROOT / "integrations" / "grok-bot" / "SKILL.md").read_text()
    assert text.startswith("---\nname: ")
    assert "Never send" in text and "Stop." in text and "owner's own computer" in text
    steps = re.findall(r"^\d\. \*\*(\w+)\.\*\*", text, re.MULTILINE)
    assert steps == ["Recall", "Draft", "Check", "Stop"]
    assert "UNKNOWN" in text
    for tool in ("search_notes", "get_note"):
        assert tool in text and tool in notes_server.READ_ONLY_TOOLS
