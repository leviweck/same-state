# Agent instructions for Same State (package and command: `plm-kit`)

Same State is a local, CPU-friendly personal-notes kit that needs no GPU: hybrid retrieval over your own notes
(recall), optional answers in your style (voice lite) from a local model, and read-only interfaces so agents can use
it while you keep the final say. It is designed to work with Grok (an optional xAI API answer backend), Grok Bot
(agents that draft, never send) and Cursor (building and modifying the kit). Using it with anything else is a user
modification and unsupported. Name those products only to describe compatibility: no logos, no "official", and no
wording that suggests endorsement or partnership.

## Architecture map
- `plm_kit/docs.py` chunking (heading-aware, 800/150 windows); `bm25.py` SQLite FTS5; `dense.py` ONNX embeddings;
  `fusion.py`/`pipeline.py` RRF fusion, stale-copy demotion (`stale.py`), optional newest-copy step (`dedupe.py`),
  optional reranker (`rerank.py`), the not-in-notes gate (block | warn | off).
- `plm_kit/store.py` index in the user cache (never in the repo). `paths.py` refuses a cache inside the repo.
- `plm_kit/answer.py` pluggable answer backends; built-ins in `plm_kit/answer_backends/`. Default `auto` (no GPU):
  Ollama if it's running, else llama-cpp-python if installed, else clear install steps. `transformers_peft` (CUDA)
  is opt-in only. Also `xai`,
  which is off by default and sends excerpts off the machine only with `generation.allow_off_machine: true`,
  a one-time consent and a per-run cost cap).
- `plm_kit/voice_lite.py` few-shot style + 12-gram copy guard. `plm_kit/safety.py` the outbound draft check.
- `plm_kit/notes_server.py` read-only MCP server (`plm-kit mcp`): search_notes, get_note, draft_check, copy_check,
  voice_examples, status. It must never gain a send, write, delete or network tool.
- `plm_kit/cli.py` thin argument parsing only: put logic in a module and keep CLI edits minimal.
- Defaults that stay as they are unless the maintainer says otherwise: retrieval mode `fused`, reranker off
  (quality preset opt-in), newest-copy off, answer backend `auto` (local, no GPU; CUDA opt-in),
  `generation.allow_off_machine: false`, no network except explicit downloads/sync.

## Safety gates (all must pass before a commit)
1. `python -m pytest -q` (tests that need downloaded models skip without them).
2. `ruff check .`
3. `gitleaks dir . --no-banner --redact` (or `gitleaks detect --no-banner --redact` once you have commits).
4. License gate: `pip-licenses --from=mixed --format=json > /tmp/licenses.json && python scripts/check_licenses.py /tmp/licenses.json`.
5. Content rule: `python -m pytest -q tests/test_content_rules.py`. Keep private names in
   `config/denylist.local.txt` (gitignored) or a file named by `PLM_KIT_DENYLIST`, never in the repo.

## Data rules
- Synthetic data only in the repo: sample notes, eval sets, style examples, tests. Never real notes, emails,
  contacts, voice data, model weights, indexes, keys or tokens. No real personal data in eval sets, ever.
- Secrets come from environment variables (e.g. `XAI_API_KEY`), never config files or code.
- New dependencies: pin in `requirements.txt` and `requirements.lock`, add to `THIRD_PARTY_NOTICES.md`, run the
  license gate.

## Drafts and actions
The kit never sends anything. Agents may draft; the owner approves every outbound message in their own app.
See `integrations/grok-bot/SKILL.md`.

## Cursor Cloud specific instructions
- The image provides `/opt/plm-kit-venv`. `python`, `pytest`, `ruff`, `plm-kit`, `pip-licenses`, and `gitleaks` are on `PATH`. `python` is that virtualenv, with CPU PyTorch (`torch==2.8.0+cpu` from the CPU wheel index).
- Dependencies come from `requirements.lock` (CPU index `https://download.pytorch.org/whl/cpu` plus PyPI), then `requirements-dev.txt` and `requirements-mcp.txt`, then an editable install of this checkout. Do not create a second virtualenv in the repo.
- No process starts on boot. Search and the starter scorecard are local commands: `plm-kit selftest` (keyword search until an embedding export exists) and `python -m eval.run_eval` (BM25 until `PLM_KIT_EMBED_ONNX` is set).
- Model files are not in the image. `plm-kit download` is explicit and writes only to the cache (`PLM_KIT_CACHE`, default `~/.cache/plm-kit`), never the repo. Tests that need those files skip.
