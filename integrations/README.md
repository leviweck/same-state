# Integrations

Same State (package and command: `plm-kit`) recalls what you know from your own notes on your own computer, with
no GPU needed. These files describe how it works with three tools. Using it with any other tool is a user
modification and unsupported.

| Path | What it is |
|---|---|
| `grok-bot/SKILL.md` | Instructions for a Grok Bot: recall, draft, check, then stop for your yes. The Bot runs `plm-kit` on your computer with per-command approval and never sends anything |
| `cursor/README.md` | How Cursor uses `AGENTS.md`, `.cursor/rules/` and `.cursor/mcp.json` (the read-only notes server, `plm-kit mcp`) |

The optional xAI answer backend (`generation.backend: xai`, off by default) is described in the main README and
`PRIVACY.md`.

---
Grok, Grok Bot, xAI and Cursor are trademarks of their respective owners; Same State is not affiliated with or endorsed by them.
