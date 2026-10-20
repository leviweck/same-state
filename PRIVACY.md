# Privacy

Support contact: samestateguide@outlook.com (Licensor: Levi Eck)

plm-kit runs on your machine. It has no server, no account, no analytics and **no telemetry**. The kit never
sends your notes, questions, answers or usage anywhere.

## What touches the network, and when
Only these commands, and only when you run them:

| Command | Connects to | What it fetches | Sends |
|---|---|---|---|
| `plm-kit download` | Hugging Face (`huggingface.co` and its file CDN) | The pinned model revisions you confirm: the embedding model, the optional reranker, and (with `--include-generator`) Qwen3-8B | Standard HTTP requests for those files. No notes or queries. An `HF_TOKEN` in your environment is sent if you set one (not needed for these public models) |
| `plm-kit onedrive login` | Microsoft identity platform (`login.microsoftonline.com`) | A sign-in token for **your own** app registration, scope `Files.Read` | Your sign-in, handled by Microsoft |
| `plm-kit onedrive sync` | Microsoft Graph (`graph.microsoft.com`) | The `.md`/`.txt` files in the folders you select (GET only) | Read requests only. Nothing is written to OneDrive |
| `plm-kit download --components gguf` | Hugging Face | One pinned GGUF answer model file (sha256-checked) | Standard HTTP requests for that file |
| `pip install ...` | PyPI / the PyTorch index | Pinned Python packages | Standard pip traffic |
| `ask --answer` / `selftest --answer` with `generation.backend: auto` (the default) | **Your own machine only**: Ollama's `/api/version` at the loopback address (`127.0.0.1:11434` by default) | Whether Ollama is running | **No notes.** A local check that bypasses proxies. Only this check is automatic; another host is used only if you set both `generation.local.allow_remote` and `generation.allow_off_machine` |
| `ask --answer` / `selftest --answer` with `generation.backend: ollama` or `llamacpp_server` | **Your own machine only**: Ollama (`127.0.0.1:11434`) or llama-server (`127.0.0.1:8080`) | The answer | **The prompt, which contains excerpts of your notes** (and your voice lite examples, if on). The kit refuses any non-loopback address (`localhost` must resolve to loopback) unless you set **both** `generation.local.allow_remote: true` and `generation.allow_off_machine: true`; a server elsewhere then counts as sending your notes off this machine. Requests never go through a proxy (`http_proxy` / `HTTPS_PROXY` are ignored). Ollama and llama.cpp are separate programs you install; their own update checks or model pulls are theirs, not the kit's |
| `ask --answer` / `selftest --answer` with `generation.backend: xai` (**off by default**) | **xAI's API over the internet**: `api.x.ai` (or `us.api.x.ai` with `generation.xai.region: us`) | The answer and token usage | **Your question, excerpts of your notes and (if voice lite is on) your style examples**, plus your API key from `XAI_API_KEY`. Your prompt and the retrieved note excerpts go to xAI and are handled under xAI's terms and policies, not this kit's. Only with `generation.allow_off_machine: true` AND a one-time consent on this computer. Requests use `store: false`. Nothing is sent if a request's estimated cost (including a reasoning allowance) would exceed the per-run (one command, e.g. a whole `selftest --answer`), daily or monthly cost cap. See "Hosted answers (xAI)" below |
| `plm-kit mcp` (read-only notes server) | **Nothing.** It talks over stdio to the one program that started it (e.g. Cursor). It opens no port and makes no network calls | Nothing | What its tools return (excerpts, draft-check results, and your style lines only if `integrations.mcp.expose_voice_examples: true`) goes to the calling agent, and from there to **that agent's model provider**. See "Agents (Cursor, Grok Bot)" below |
| `ask --answer` / `selftest --answer` with a **third-party** answer-backend plugin (`generation.backend_modules` or an installed `plm_kit.answer_backends` entry point) that declares `sends_off_machine` | That plugin's service | The answer | **Your question and excerpts of your notes** (and your voice lite examples, if on). Refused unless you set `generation.allow_off_machine: true`, the same switch the built-in `xai` backend needs. Unlike `xai`, the kit can't verify a third-party plugin: it relies on the plugin declaring `sends_off_machine` honestly and adds no consent prompt or cost cap of its own, so only install backends you trust |

One case the kit can't detect: if you set `generation.ollama.model` to an Ollama "cloud" model, Ollama itself sends your question and note excerpts to Ollama's servers under Ollama's own terms. The default `qwen3:4b-instruct` runs locally; use a local model if you want everything to stay on your computer.

`index`, `ask` (including `--answer`), `eval`, `selftest`, `copy-check`, `health` and `draft-check` work offline: the embedding model and
reranker load from your local ONNX export, and the generator loads with `local_files_only=True`. `download` sets
`HF_HUB_DISABLE_TELEMETRY=1` for its own process. For a hard guarantee, also set `HF_HUB_OFFLINE=1` after
downloading.

Some libraries the kit uses have their own usage reporting. The kit turns it off for its own process: it sets `HF_HUB_DISABLE_TELEMETRY=1` for Hugging Face and `ORT_DISABLE_TELEMETRY=1` for Microsoft's ONNX Runtime (the search engine's runtime) before either library loads. If you import these libraries in your own scripts outside the kit, set those variables yourself.

## What is stored, and where
Everything lives in your user cache (`$PLM_KIT_CACHE`, default `~/.cache/plm-kit`) unless you change the path.
The kit refuses to put any of these inside its own repo folder.

| Data | Location | Contains |
|---|---|---|
| Downloaded models + local ONNX exports | `<cache>/<model>-src`, `<cache>/<model>-onnx`, `<cache>/qwen3-8b-src`, `<cache>/gguf/` | Public model files |
| Voice lite examples | Wherever `generation.voice_lite.examples_file` points (keep it outside the repo) | 5-8 lines you wrote. Read only |
| Notes index | `<cache>/index/` (or `data.index_dir`) | **The text of your notes** (chunks), their embeddings, and a manifest of file hashes |
| Synced OneDrive files | `<cache>/sources/onedrive/` (or `sources.onedrive.sync_dir`) | Copies of the files you selected, plus a sync-state file (item ids and eTags) |
| OneDrive token cache | OS user cache dir: `~/.cache/plm-kit/onedrive-msal-cache.json` (Linux), `~/Library/Caches/plm-kit/` (macOS), `%LOCALAPPDATA%\plm-kit\` (Windows), or `PLM_KIT_TOKEN_CACHE`. File mode 0600 | Access and refresh tokens for `Files.Read` |
| xAI consent record (only if you use the `xai` backend) | `<cache>/consent/xai-v1.json` | That you consented, and when. No key, no text |
| xAI spend ledger (only if you use the `xai` backend) | `<cache>/spend/xai-ledger.json` | Local dates and dollar amounts only (`{"days": {"YYYY-MM-DD": usd}}`), plus, on days when an answer cost more than its estimate, a count per date (`{"overshoots": {"YYYY-MM-DD": n}}`). Kept for about 400 days. No prompts, answers, questions or keys |
| xAI spend ledger lock (only if you use the `xai` backend) | `<cache>/spend/xai-ledger.json.lock` | Nothing: an empty file used only as a lock handle, so two plm-kit processes can't update the ledger at the same time |
| Your notes | Wherever `data.docs_dir` points | Read only; never modified |

Nothing is logged to disk. `eval --json` writes only where you tell it to. Error messages don't include note text.

## Plain-text storage

The index and any synced OneDrive copies are stored as plain text in the cache. Keep your device encrypted and locked (FileVault, BitLocker or LUKS), and delete the cache when you no longer need it (see "Deleting your data").

## Deleting your data
```bash
plm-kit onedrive logout                    # deletes the token cache file
rm -rf ~/.cache/plm-kit/index              # the notes index (rebuilt by `plm-kit index`)
rm -rf ~/.cache/plm-kit/sources/onedrive   # synced OneDrive copies
rm -rf ~/.cache/plm-kit                    # everything: models, exports, index, synced files
```
Use your `PLM_KIT_CACHE` / configured paths instead if you changed them. Hugging Face's own download cache
(`~/.cache/huggingface`) is only used if you ran Hugging Face tools yourself.

## OneDrive access (read-only)
- Scope requested: **`Files.Read`** (delegated: only what your signed-in account can read). MSAL also requests
  `openid`, `profile` and `offline_access` (refresh without signing in again). No write scopes; the kit refuses a
  token that has one, and tests fail if any write scope or write HTTP method appears in the code.
- It uses **your** app registration; the kit ships no client or tenant ids. See `docs/ONEDRIVE.md`.
- Revoke at any time: `plm-kit onedrive logout`, then remove the app's consent in your Microsoft account
  (<https://account.live.com/consent/Manage>) or My Apps, or delete the app registration.

## Your responsibility

Only connect accounts and folders you're allowed to copy to your own computer. For a work or school OneDrive, check your employer's policies first. Synced files may contain other people's information; you're responsible for handling it lawfully.

## Third-party services

Microsoft's and Hugging Face's own terms and privacy policies govern those services. Your OneDrive app registration falls under the [Microsoft APIs Terms of Use](https://learn.microsoft.com/en-us/legal/microsoft-apis/terms-of-use).

## Hosted answers (xAI)
Off by default. With `generation.backend: xai`, `generation.allow_off_machine: true` and a one-time consent
(`python -m plm_kit.answer_backends.xai --consent`, or answer the prompt; `--revoke` removes it), each answered
question sends the prompt (question + excerpts, plus your style examples if voice lite is on) to xAI under your own
account and key, and xAI handles it under xAI's terms and policies, not this kit's. The kit sends `store: false`, so xAI keeps no conversation state for later continuation. xAI's own
retention still applies: its docs say API requests and responses are stored encrypted for 30 days for abuse auditing
and aren't used for training without your permission (https://docs.x.ai/developers/faq/security). **Zero data
retention** is a team-wide setting a team admin turns on in the xAI Console; it is not an API parameter, so the kit
can't switch it on for you. Don't opt in to any xAI data-sharing credit program with personal notes. The consent
record is a small file in `<cache>/consent/`. Your key is read from the environment and never stored or logged.

Spending limits, all checked **before** anything is sent, using an estimate for the request (prompt size, the
longest visible answer, and `generation.xai.reasoning_allowance_tokens` for the model's reasoning): `generation.xai.max_cost_usd` per run (one command's process, across all its answers), `max_cost_usd_per_day` (default $2) and the optional
`max_cost_usd_per_month` per computer. The daily and monthly totals live in the spend ledger above. Every ledger
update takes a file lock (fcntl on macOS and Linux, msvcrt on Windows); if no lock is available, or it can't be
taken within 60 seconds, the kit refuses to send rather than update the ledger unlocked. If the lock fails after an answer has already come back, the kit reports an error saying the request was billed, and the ledger keeps that request's estimated amount, which can be lower than the real cost. Each request
reserves its estimate before sending and is corrected to the real cost afterwards, so a failed request keeps its
estimated amount for that day. xAI doesn't count reasoning against the visible-answer limit and documents no bound
for it, so one request can cost more than its estimate and go slightly past a cap; the real cost is still recorded
and the overshoot is flagged. Deleting the ledger resets the totals. The limits are the kit's own estimate from
`generation.xai.prices`; xAI's prepaid balance is the hard limit, so keep prices current.

The US regional endpoint (`region: us`) keeps inference and retained request data in the United States, but it
serves only `grok-4.7` and `grok-4.6`; the kit rejects other models with `region: us`
(https://docs.x.ai/developers/advanced-api-usage/regions).

## Agents (Cursor, Grok Bot)
`plm-kit mcp` and the Grok Bot skill (`integrations/grok-bot/SKILL.md`) never send anything and never write to your
notes. But anything an agent reads through them enters that agent's context and goes to its model provider. Where
Grok Bot runs and how it handles data are governed by its provider's terms; see
https://docs.x.ai/grok-bot/approvals-security-and-privacy and https://cursor.com/docs/grok-bot. Keep the notes on your own computer: let a Grok Bot run plm-kit there
with per-command approval rather than copying your notes or index to a cloud computer. Turn on Cursor's Privacy Mode.

## Generated answers and drafts
`ask --answer` runs the model locally: on your GPU (`transformers_peft`), or on your CPU / Apple Silicon through
Ollama or llama.cpp on the same machine. Voice lite examples stay in your file and in the local prompt; the
copy guard keeps them out of answers, and `copy-check` never prints them. `draft-check` only scans text you give it, locally. A human
reviews every draft before anything is sent; the kit sends nothing.
