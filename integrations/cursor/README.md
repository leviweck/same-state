# Same State in Cursor

Same State (package and command: `plm-kit`) is designed to be built and modified with Cursor.

- `AGENTS.md` (repo root): architecture map, safety gates and data rules. Cursor applies it automatically.
- `.cursor/rules/*.mdc`: `safety-gates.mdc` (always applied), `architecture.mdc` (plm_kit/, tests/, config/),
  and `eval-data.mdc` (eval/, starter_kit/, sample_data/). Rules must use the `.mdc` extension.
  Docs: https://cursor.com/docs/context/rules
- `.cursor/mcp.json`: starts the read-only notes server (`python -m plm_kit.cli mcp`, stdio) so Cursor's Agent can
  call `search_notes`, `get_note`, `draft_check`, `copy_check`, `voice_examples` and `status`. Cursor asks before
  each MCP tool call by default. Docs: https://cursor.com/docs/context/mcp

## Setup
1. Install the kit's requirements in the Python environment Cursor uses (`pip install -r requirements.txt -r requirements-mcp.txt`, or at
   least `pyyaml`, `numpy` and `mcp`). `python` on Cursor's PATH must be that environment; otherwise change
   `"command"` in `.cursor/mcp.json` to its full path. Run `plm-kit index` once.
2. Optional: set `PLM_KIT_CONFIG` to your own config file (outside the repo) before starting Cursor.
3. Open the repo folder in Cursor. Cursor reads project MCP servers from `.cursor/mcp.json` in the project
   (global ones from `~/.cursor/mcp.json`). Docs: https://cursor.com/docs/context/mcp ("Configuration locations").
4. Turn the server on or off from **Customize** in the sidebar: find `plm-kit-notes` and use its toggle. Disabled
   servers don't load or appear in chat. Docs: same page, "Using MCP in chat" and FAQ "Can I temporarily disable an
   MCP server?". Whether Cursor asks you to trust a project's server the first time it loads may vary by version.
5. In chat, Cursor asks for approval before each MCP tool call by default (same page, "Tool approval"). Keep that
   on. If something fails, open the Output panel and choose "MCP Logs".
6. Team plans: an admin's MCP allowlist can block project servers (same page, "MCP Allowlist"). Ask your admin to
   allow the command `python -m plm_kit.cli mcp` if needed.

## Privacy
Whatever a tool returns (excerpts of your notes, and your style examples if you turned
`integrations.mcp.expose_voice_examples` on) goes into the Agent's context, and so to Cursor's model providers.
Turn on Privacy Mode in Cursor settings. Leave the notes server disabled when you don't need it.

---
Grok, Grok Bot, xAI and Cursor are trademarks of their respective owners; Same State is not affiliated with or endorsed by them.
