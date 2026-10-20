# Starter test kit (`plm-kit selftest`)

Know the kit works on your machine in a few minutes, on CPU, before you point it at your own notes.

```bash
plm-kit selftest                 # keyword search right after install (seconds)
plm-kit download --components embed && plm-kit selftest     # + the default fused mode and the not-in-notes gate
plm-kit selftest --answer        # + three answers and voice lite via your no-GPU backend (docs/NO_GPU.md)
plm-kit selftest --json card.json
```
It prints a scorecard and exits 0 if every check passes, 1 if any fails. It indexes the sample notes **in memory**
and doesn't touch your notes, index or cache, and uses no network (except `--answer` talking to your local model).

## What's here (all synthetic)
- `sample_notes/`: 33 short notes of a **fictional** one-person design and print studio (pricing, invoicing,
  taxes, suppliers, clients, routines). The clients and every detail are made up. It includes the kinds of copies
  real note folders have:
  - stale copies in `archive/` and `old/` (older prices, an older deposit rule, an old schedule),
  - machine-made copies: a web-page export (`notes-export/*.html.md`) and a chat summary
    (`chat-summaries/*chat-summary*`) with slightly wrong facts.
- `golden_questions.jsonl`: 40 questions. 32 have a known answer note (`relevant`); 7 of those also list the stale
  copies that must rank **below** the live note (`stale`); 8 are not in the notes (`answerable: false`), most of
  them near misses. `answer_contains` is used only by `--answer`, for information.
- `thresholds.yaml`: the pass/fail limits, per mode, with the maintainer's measured values.
- `style_examples.txt`: 6 made-up lines in the fictional owner's voice for the voice lite check.

## The checks
| Check | Meaning |
|---|---|
| recall@3 | Share of answerable questions whose answer note is in the top 3 |
| mrr | Mean reciprocal rank of the answer note (1.0 = always first) |
| stale_demoted | Share of the 7 stale-copy questions where the live note outranks every archived or exported copy |
| nin_caught | Not-in-notes questions caught by the gate (fused mode only). A partial catch is expected: the gate is a trade-off |
| false_gated | Answerable questions the gate wrongly blocked (fused mode only) |
| answers_non_empty / style_copies_in_output | `--answer` only: every answer came back, and no voice lite answer contains a copied style example |

Measured on the maintainer's 8-core CPU box (no GPU): keyword-only recall@3 1.0, MRR 0.943, stale 7/7, about
0.1 s; fused recall@3 1.0, MRR 0.984, stale 7/7, gate 2/8 caught and 0/32 false-blocked, about 4 s including
model load. `--answer` with the 0.5B smoke-test model added about 5-10 s on each backend. Bigger models take longer.

## Your own notes
These numbers say the install works, not how well it will do on your notes. For that, write your own questions
and run `plm-kit eval --docs /path/to/notes --gold your-questions.jsonl` and `plm-kit calibrate` (main README).
