# Same State: a personal language model kit

A local, CPU-friendly personal-notes retrieval kit: hybrid BM25 + dense retrieval fused with Reciprocal Rank Fusion (RRF), an optional pluggable reranker that is **off by default**, and a local LLM for answers.

> Status: v1.0.5. Same State is a working name (no ™/®). The package and command are `plm-kit`.

What's a Human AI Brain? Here's my definition: https://samestateguide.com/human-ai-brain

## What ships / what doesn't
- Ships: code, config, tiny **synthetic** sample docs, a synthetic eval set.
- Never ships: model weights, LoRA adapters, search indexes, personal notes, emails, contacts, voice data, secrets, or any employer material.

## Quick start: confirm it works (CPU only)
```bash
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\activate
pip install pyyaml==6.0.3 numpy==2.5.4 && pip install -e .   # enough for the keyword-search selftest
plm-kit selftest                  # sample notes + golden questions -> PASS/FAIL scorecard (seconds)
```
Then the full default (fused) mode: install `requirements.txt` (CPU PyTorch: add
`--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`), export the embedding
model as described under "Models" (`plm-kit download --components embed`, ~0.6 GB, export tools in their own venv
from `requirements-export.txt`), and run `plm-kit selftest` again: it then also scores the fused mode and the
not-in-notes gate. Download and export time depend on your connection and CPU.
**Answers, no GPU needed (default):** install [Ollama](https://ollama.com) and run `ollama pull qwen3:4b-instruct`, or
`pip install -r requirements-local.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` (prebuilt
CPU wheels; PyPI has only the source) and `plm-kit download --components gguf`. Then `plm-kit ask "..." --answer`
and `plm-kit selftest --answer` just work: `generation.backend: auto` uses Ollama if it's running, else
llama-cpp-python, and otherwise tells you how to install either (see "No-GPU answers").
The selftest uses the bundled **synthetic** notes of a fictional studio (`starter_kit/`), runs in memory and doesn't
touch your notes or index. Details and the meaning of each check: `starter_kit/README.md`.

## Quick start (dev)
Python 3.11+.
```bash
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env            # dummy values only
python -m pytest -q             # blend fake-scorer tests + disabled-reranker no-load tests
python -m eval.run_eval          # synthetic golden set; scores recall@k, hit@k, MRR, not-in-notes
```

## CLI
Install with `pip install -e .` (or run `python -m plm_kit.cli`):

| Command | What it does |
|---|---|
| `plm-kit download [--components embed,reranker] [--include-generator] [--yes]` | **You trigger it explicitly.** It prints a license notice, asks you to type `yes`, fetches the pinned revisions from Hugging Face into your cache (`$PLM_KIT_CACHE`, default `~/.cache/plm-kit`, never inside the repo), then exports ONNX locally. Qwen3-8B (~16 GB) is fetched only with `--include-generator`; `--components gguf` fetches the pinned no-GPU answer model (`generation.local.model`, sha256-checked). Needs `requirements-export.txt` in that environment. |
| `plm-kit index [--docs DIR]` | Builds or refreshes the index of your notes in your cache. It rebuilds when the notes, the chunking/skip settings or the embedding model change. |
| `plm-kit ask "question" [--k 5] [--mode dense\|fused\|fused+rerank] [--preset quality]` | Shows excerpts from your notes, or "Not in your notes". `--mode` overrides `retrieval.mode` for this query; `--preset quality` = fused+rerank with top_n 20, weight 1.0 (about 2 s/query on CPU). `--answer` generates an answer with `generation.backend` (or `--backend`). The default `auto` needs no GPU (Ollama if running, else llama-cpp-python); `ollama`, `llamacpp_server` and `llamacpp_python` can be picked directly; `transformers_peft` is the opt-in CUDA GPU path. `--voice-lite` answers in your style from 5-8 example lines (no training). |
| `plm-kit eval [--modes bm25,dense,fused,fused+rerank] [--newest-copy off\|demote\|drop] [--docs DIR --gold FILE] [--fixed-threshold] [--json OUT] [--no-grid]` | Runs the golden-set eval per mode (default: the synthetic set; every mode your downloaded models allow; `--fixed-threshold` gates at your configured cutoff instead of tuning one; see `eval/README.md`). |
| `plm-kit calibrate QUESTIONS.jsonl` | Picks a not-in-notes **trade-off** from **your own** labelled questions (`{"query": "...", "answerable": true}` per line): prints the score spread, and how many out-of-notes questions a threshold catches and how many answerable ones it falsely blocks. It can't fix overlapping scores. Local only; changes nothing. |
| `plm-kit health` | Shows config, model and index status. |
| `plm-kit selftest [--answer] [--backend ...] [--json OUT]` | **Starter test kit.** Indexes the bundled synthetic notes in memory, runs the golden questions (including not-in-notes ones) and prints a PASS/FAIL scorecard against `starter_kit/thresholds.yaml`; exit 1 on any FAIL. `--answer` also checks your no-GPU backend and the voice lite copy guard. |
| `plm-kit copy-check FILE [--examples F] [--n 12]` | Flags any run of 12+ words (or a whole example) copied from your voice lite examples. Reports positions only, never the example text. |
| `plm-kit draft-check FILE` | Flags emails, phone numbers, card-like numbers, token-like strings and words from `config/safety_words.txt` in an outbound draft. Advisory only; you still review before sending. |
| `plm-kit mcp [--docs DIR]` | Read-only notes server for agents (Model Context Protocol over stdio): `search_notes`, `get_note`, `draft_check`, `copy_check`, `voice_examples` (off unless you allow it), `status`. No send, write or network tools. Optional: `pip install -r requirements-mcp.txt` first. Used by Cursor (`.cursor/mcp.json`). See `integrations/`. |
| `plm-kit onedrive login\|sync\|status\|logout` | Optional, **read-only** OneDrive source (Microsoft Graph, `Files.Read`, your own app registration). `sync` copies the folders you select into your cache and reindexes. See `docs/ONEDRIVE.md`. |

Only `download` (models), `onedrive login` / `sync` (your own files, read-only) and, if you explicitly turn it on, the hosted `xai` answer backend touch the internet. The no-GPU backends talk to a model server on your own machine (127.0.0.1 only, unless you allow otherwise). See `PRIVACY.md` for what is downloaded, what is stored where, and how to delete it. The kit has no telemetry of its own, and it turns off the built-in usage reporting of Hugging Face and ONNX Runtime (see `PRIVACY.md`).

## Integrations: Grok, Grok Bot, Cursor
Same State recalls what you know on prompt, with a local model and no GPU by default. It is designed to work with
the three tools below to carry out what needs doing, with you approving anything that leaves your hands. Using it
with other tools is a user modification and unsupported. Index of the files: `integrations/README.md`.
- **Grok (xAI API)**: an optional hosted answer backend instead of the default local `auto` backend, `generation.backend: xai`. It's off by default. It needs
  `generation.allow_off_machine: true`, a one-time consent and your own key in `XAI_API_KEY`. It uses the Responses
  API with `store: false`. It refuses before sending if a request's estimated cost (with a reasoning allowance) would exceed the per-run cap
  (`max_cost_usd`, one command with all its answers), the daily cap (`max_cost_usd_per_day`, default $2) or the optional monthly cap, tracked in a
  small local ledger of dates and amounts. `region: us` serves only grok-4.7 / grok-4.6. Prices live in
  `generation.xai.prices`; check https://docs.x.ai/developers/pricing. Privacy: `PRIVACY.md`.
- **Grok Bot**: `integrations/grok-bot/SKILL.md`. Recall, draft, check, then stop for your yes. The Bot runs plm-kit
  on your own computer with per-command approval and never sends anything itself.
- **Cursor**: `AGENTS.md`, `.cursor/rules/`, and `.cursor/mcp.json` (the read-only notes server). See
  `integrations/cursor/README.md`.

Grok, Grok Bot, xAI and Cursor are trademarks of their respective owners; Same State is not affiliated with or endorsed by them.

## Config
`config/default.yaml` holds the defaults. `config.example.toml` lists every setting with comments and dummy values: copy it outside the repo, edit it, and pass `--config /path/to/it.toml` (or set `PLM_KIT_CONFIG`). YAML and TOML both work. The embedding model is swappable in config (`retrieval.dense.custom`). Changing it triggers a reindex, so check the new model's license and training data first.

## Retrieval
Details, settings and known failure cases: `docs/RETRIEVAL.md`.
- **Mode** (`retrieval.mode`, default `fused`): `dense` (embeddings only), `fused` (BM25 + dense with RRF), or `fused+rerank` (fused, then the optional reranker). `plm-kit ask --mode ...` overrides the config for one query. `dense` needs the downloaded embedding model; `fused` falls back to BM25 alone until it's there. `fused+rerank` falls back to `fused` (with a note) if the reranker isn't downloaded, `PLM_KIT_RERANKER=off` is set, or `top_n`/`weight` is 0.
- **Quality preset** (opt-in): `retrieval.preset: quality` or `--preset quality` = `fused+rerank` with `top_n: 20`, `weight: 1.0`. Roughly **2 s per query on CPU** (the maintainer's test). The reranker stays off by default.
- **Chunking:** front matter stripped; split at every Markdown heading (the heading path is kept); each section windowed at 800 chars with 150 overlap (`data.chunking`). Embedded text is `title > heading path`, newline, chunk. Notes over 2 MB and skip-listed files/folders (`.git`, trash, `.obsidian`, `.vscode`, `*.bak`, names containing `backup`, ...) aren't indexed. Embedding and reranker inputs are also capped at 512 tokens.
- **BM25** keyword search with SQLite FTS5 `bm25()` over title, heading path and text (`plm_kit/bm25.py`). User text is turned into quoted OR-terms, so FTS syntax can't be injected.
- **Dense** search with the pinned `granite-embedding-english-r2`, exported to ONNX and run on CPU via fastembed (`plm_kit/dense.py`). Its model card specifies no query/document prefixes, so none are added. Vectors are normalised.
- **Fusion:** Reciprocal Rank Fusion with k=60, then **stale-copy demotion** (`retrieval.stale`): notes in an `archive`/`archived`/`backup`/`backups`/`old` folder x0.5, generic "derived" copies (exports, chat summaries) x0.6, your optional "owner" folders x1.2. Nothing is dropped. Then at most 2 chunks per note (`per_note_cap`).
- **Optional reranker** (off by default). The blend is `1/(k+fused_rank) + weight/(k+ce_rank)`. If you turn it on, use the quality preset values (`top_n: 20`, `weight: 1.0`).
- **Newest-copy step** (experimental, off by default; we suggest leaving it off): `retrieval.dedupe.newest_copy: demote` groups near-duplicate notes and, when several copies are retrieved, keeps the one that is clearly newest by file modification time (ties within `tie_tolerance_s`, default 60 s, are left alone). **Warning:** synced or copied folders (OneDrive, Dropbox, git checkouts, restores) reset file times. In the maintainer's real-notes test it didn't help and once demoted the right answer. `drop` exists but is not recommended. See `docs/RETRIEVAL.md`.
- **Owner/derived lists:** add your own globs or regexes in `retrieval.stale` or in a separate file (`retrieval.stale.lists_file`, example: `config/stale.example.yaml`). Owner beats derived; the archive-folder rule stacks on either.
- **Not-in-notes gate:** if the best dense cosine falls below `not_in_notes_threshold` (default **0.78**), the answer is "not in your notes" (`not_in_notes_gate: block`, the default). BM25-only setups fall back to "any keyword hit". **The gate is unreliable on real notes** (see below). We suggest `not_in_notes_gate: warn` (results are shown with a "weak match" warning) or `off`.
- Eval: see `eval/README.md`.

### What testing on real notes showed
The synthetic set (fictional notes) ranked dense-only first. **Testing on real notes reversed that.** The maintainer's internal test (107 questions over a real notes collection, CPU), re-run with the heading-aware chunking and stale-copy demotion, found:

| Setting | MRR | hit@1 | Speed |
|---|---|---|---|
| dense | 0.551 | 0.393 | fast |
| **fused (default)** | **0.647** | **0.505** | fast |
| quality preset (fused+rerank, top_n 20, weight 1.0) | 0.695 | 0.551 | ~2 s/query |

Fused beat dense. The quality preset scored higher on both MRR and hit@1, but is much slower on CPU, so it stays opt-in and the default stays `fused` with the reranker off. Your notes will differ: run your own comparison (`plm-kit eval --docs ... --gold ...`).

**The not-in-notes gate:** in the same test no single cutoff separated the scores. The best dense scores of out-of-notes questions ranged 0.775 / 0.851 / 0.897 (min / median / max); answerable questions had min 0.770, median 0.889. So the two groups overlap almost completely:

| Threshold | Out-of-notes caught (of 8) | Answerable falsely blocked |
|---|---|---|
| 0.78 (default) | 1 | 1 |
| 0.82 | 3 | 4 |
| 0.84 | 4 | 11 |

Raising the threshold catches more out-of-notes questions only by blocking many real ones. Calibration can only pick a trade-off; it can't fix the overlap.

### Disabling or calibrating the not-in-notes gate
**Simplest:** set `retrieval.not_in_notes_gate: warn` (always show results, with a "weak match" warning below the threshold) or `off`. If you want to keep blocking:

1. Write a file of your own questions, one JSON object per line: `{"query": "...", "answerable": true}`. Aim for **30+ questions your notes answer** and **10+ they don't** (include near misses: topics you have notes on, but not that detail).
2. Run `plm-kit calibrate questions.jsonl`. It prints the score spread for both groups, whether they overlap, and for the current and the suggested threshold (best balanced accuracy) how many out-of-notes questions are **caught** and how many answerable ones are **falsely blocked**. That is the trade-off you are choosing.
3. Set `retrieval.not_in_notes_threshold` in your config. Lower it if real questions get "Not in your notes"; raise it if off-topic questions return excerpts. `plm-kit ask` prints the best dense score for every query (stderr) for spot checks.
4. If the scores overlap (`"scores_overlap": true`), no threshold is right; prefer `warn` or `off`. Re-run after large changes to your notes or a model change. The questions file is yours; keep it outside the repo.

## Serving
**No GPU needed.** Search, excerpts and generated answers all run on CPU (or Apple Silicon) by default. A GPU is only
used by the opt-in `transformers_peft` backend.

### No-GPU answers (Ollama / llama.cpp, CPU or Apple Silicon)
The default `generation.backend: auto` tries **Ollama** (at `generation.ollama.host`, model `qwen3:4b-instruct`),
then **llama-cpp-python** (`requirements-local.txt` + `plm-kit download --components gguf`), and otherwise stops with
install steps for either. To pin one, set `generation.backend` to `ollama`, `llamacpp_server` or `llamacpp_python`
(or pass `--backend`). Selftest and search without `--answer` never need a backend. Default model
for 8-16 GB machines: **Qwen3-4B** (Apache-2.0; Ollama `qwen3:4b-instruct`, or the pinned GGUF from
`plm-kit download --components gguf`). Smaller and alternative models, all Apache-2.0, are listed with their
licenses in `docs/NO_GPU.md`. You install Ollama or llama.cpp yourself; the kit only talks to them on 127.0.0.1.
Backends are plugins (`plm_kit/answer.py`: `AnswerBackend`, `register`, drop-in discovery), so new ones can be added
without touching core; `generation.backend: none` turns answers off.

### Voice lite (your style, no training)
Point `generation.voice_lite.examples_file` at 5-8 lines **you** wrote (one per line, outside the repo) and add
`--voice-lite`. The examples steer style only, never facts. A **verbatim-copy guard** flags any 12+ words in a row
copied from an example (or a whole example), regenerates once, then redacts or blocks, so your example lines are
never printed back. See `docs/NO_GPU.md`.

### Advanced, optional: CUDA GPU (transformers + peft)
Not needed for v1 and not part of the quick start. Opt in with `generation.backend: transformers_peft`.
- Answers come from `Qwen/Qwen3-8B` at a pinned revision, loaded with **transformers + peft** in 4-bit (bitsandbytes) on a CUDA GPU (`plm-kit download --include-generator` first).
- Advanced: `adapter_dir` can point at a LoRA adapter **you** trained yourself (peft format). Voice training is not a v1 feature and the kit ships no training tool, adapters or weights.
- LoRA adapters are served only by this backend; the no-GPU backends run the base model.
- Dependencies are pinned exactly: `requirements.txt` (direct) and `requirements.lock` (full transitive set).

## Models (downloaded by you, never bundled)
- Dense embeddings: `ibm-granite/granite-embedding-english-r2` (Apache-2.0), pinned revision in `plm_kit/models.py`.
- Optional reranker, **off by default**: `ibm-granite/granite-embedding-reranker-english-r2` (Apache-2.0), pinned revision.
- No-GPU answer models (optional): Qwen3-4B (default), Granite 4.0 Micro, Qwen2.5-1.5B/0.5B Instruct GGUF, all Apache-2.0, pinned revisions + sha256 in `plm_kit/models.py` (`docs/NO_GPU.md`).
- To set up, **on your own machine**, in its own venv (for example `.venv-export`; its torch pin conflicts with
  `requirements.txt`): `pip install -r requirements-export.txt`, then `python scripts/export_onnx.py granite-embed-r2 ~/.cache/plm-kit` (and `granite-r2` for the reranker). The script downloads the pinned revision from Hugging Face and runs the ONNX export locally. Nothing is converted or hosted by the kit. Point `onnx_dir` in `config/default.yaml` at the export. The script refuses to write inside the repo.
- To use the reranker: `retrieval.mode: fused+rerank` (or `plm-kit ask --mode fused+rerank`). `PLM_KIT_RERANKER=off` forces it off. See `docs/RERANKER.md`.

## Packaging a release
The maintainer builds release archives with a packaging script that is not part of this download: it uses `git archive` on an allowlist of committed files (never a folder copy), rejects weight/index/data/.env file types, runs gitleaks on the result, and writes a sha256. Then read every file in the extracted archive before publishing it.

## License
This repository is licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).

Support contact: samestateguide@outlook.com (Licensor: Levi Eck)

**Optional and third-party components, including every model you download, are subject to their own licenses.** See `THIRD_PARTY_NOTICES.md`.

Third-party models are downloaded by you from Hugging Face and are licensed to you by their owners under their own terms, not by this kit.
