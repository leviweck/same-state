---
name: use-my-notes
description: Recall from the owner's own notes with Same State (command plm-kit) on the owner's computer, draft in the owner's voice, check the draft, then stop for the owner's yes. Never sends anything.
---

# Use my notes (Same State)

Use this skill when the owner asks something their notes may answer, or asks for a draft (email, message, reply,
post) that should use what their notes say. Same State runs on the owner's computer as the command `plm-kit`,
with a local model and no GPU needed. The notes stay there. This skill only reads them.

## Rules
1. **Never send, post, submit, schedule or share anything.** Your output is a draft shown in this conversation.
   The owner sends it themselves, or tells you to send it in a separate step that goes through the app's
   approval card. Never use "Always allow" for sending.
2. Run plm-kit **only on the owner's own computer**, one approved command at a time. Never copy the notes, the
   index or the cache to the cloud computer, a shared folder or any website.
3. Use only what the commands return. If plm-kit says "Not in your notes." or prints a weak-match warning, say
   so plainly. Don't fill gaps from memory or the web without saying that's what you did.
4. Name the source notes you used (the ids plm-kit prints) under the draft.
5. Don't paste the owner's style examples into the draft. Write in the same style instead.

## Steps
1. **Recall.** On the owner's computer run:
   `plm-kit ask "<the question in the owner's words>" --k 5`
   Read the excerpts and note ids. For more of one note, run a narrower `plm-kit ask` that names it.
   (Cursor, or any agent with local MCP access, can call the `search_notes` and `get_note` tools of
   `plm-kit mcp` instead.)
2. **Draft.** Write the reply or answer from those excerpts only, in the owner's voice. If the owner turned on
   voice lite, `plm-kit ask "<question>" --answer --voice-lite` drafts locally with the copy guard. It uses the
   default `auto` backend (Ollama if running, else llama-cpp-python); no GPU or cloud model is needed.
3. **Check.** Save the draft to a temporary file on the owner's computer, then run both:
   - `plm-kit draft-check <file>`: flags emails, phone numbers, card-like numbers, token-like strings and words
     from the owner's list.
   - `plm-kit copy-check <file>`: flags wording copied word for word from the style examples (positions only).
   Fix what they flag, or explain why it should stay. Delete the temporary file afterwards.
4. **Stop.** Show the owner the final draft, the source note ids, and the check results. Ask: "Send this, edit it,
   or drop it?" Do nothing more until the owner answers. An approval covers this one draft only.

## Setup on the owner's computer (one time)
1. Install plm-kit on the owner's computer and confirm it works: `plm-kit selftest`, then `plm-kit index`
   (README "Quick start"). The index lives in the owner's user cache, not the repo.
2. Install the Grok Bot desktop app on the same computer, keep it open and signed in, and allow it to run commands
   there with **Ask every time** (the default). Settings -> General -> Bot -> Execution on Local Computer offers
   Ask every time, Always allow and Never allow. Once the computer is registered, the setting moves to
   Settings -> Computer -> Computers -> Execution on this computer. The first time a Bot asks, choose
   **Allow once**, not Always allow. A team admin can cap this setting.
   Docs: https://docs.x.ai/grok-bot/approvals-security-and-privacy ("Control access to your local computer") and
   https://docs.x.ai/grok-bot/settings-and-notifications.
   - **Supported systems:** the desktop app runs on macOS (Apple silicon and Intel), Windows (x64 and Arm64) and
     Linux (x64 and Arm64) (https://cursor.com/docs/grok-bot, FAQ "Which platforms are supported?"). The
     local-access section of the docs only mentions "the Mac or Windows computer in front of you", so on
     **Linux, running commands on your own computer may vary** by app version. If it doesn't work there, the
     plm-kit CLI and Cursor still work on Linux; only this Bot path is affected.
3. Save this file as a private skill. Ask your Bot: "Save integrations/grok-bot/SKILL.md as a skill called
   Use my notes." Skills are a shared library across your Bots (docs.x.ai/grok-bot/skills-routines-and-automations).
   - **UNKNOWN:** whether a Grok Bot skill reads this file's front matter (name, description) or only its text.
4. Add a personal Auto-review rule: "Ask first before sending any external email or message." Personal rules can
   only make the Bot stricter (cursor.com/docs/grok-bot/security).
5. Test with the bundled synthetic notes first: `plm-kit ask "how do I factory reset the router" --docs
   sample_data/docs`. Then switch to your own notes.

## What the public docs leave UNKNOWN
- Whether a Grok Bot can use a stdio MCP server running on the owner's own computer, rather than on its cloud
  computer. Until that's confirmed, this skill uses approved CLI commands. Don't point the Bot at a public URL
  for the notes server; the server is stdio-only on purpose.
- Whether a Cursor plugin installs into Grok Bot automatically.
- Whether "Always allow" rules for local commands can be scoped to read-only plm-kit commands. Keep approvals
  per command until that's documented.

---
Grok, Grok Bot, xAI and Cursor are trademarks of their respective owners; Same State is not affiliated with or endorsed by them.
