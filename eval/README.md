# Synthetic eval

`golden_synthetic.jsonl` has 80 fictional queries over `sample_data/docs` (44 fictional notes, 2 of them archive copies):
- 34 keyword-style queries and 34 paraphrase queries (little word overlap).
- 12 not-in-notes queries (15%): 7 near misses on topics the notes cover, and 5 off-topic.

Metrics (doc level): recall@1/3/5, hit@1/3/5, MRR, not-in-notes caught, answerable false-gated, and mean/p95 CPU latency per query.
With one relevant doc per query, recall@k equals hit@k.

**Split:** a deterministic, stratified 70/30 split by sha256(query id) gives 56 calibration and 24 test queries. The not-in-notes threshold is tuned on the calibration split for balanced accuracy (plain accuracy would pick "never gate"), then reported on the held-out test split.

```bash
plm-kit download                       # once: pinned models into your cache (license notice first)
PLM_KIT_EMBED_ONNX=~/.cache/plm-kit/granite-embed-r2-onnx \
PLM_KIT_RERANK_ONNX=~/.cache/plm-kit/granite-r2-onnx \
plm-kit eval --json results.json       # bm25 | dense | fused | reranker grid (top_n 6/10/20 x weight 0.25/0.5/1.0)
plm-kit eval --modes dense,fused       # only the arms you name (bm25, dense, fused, fused+rerank)
plm-kit eval --no-grid                 # without the env vars: BM25 arm only (no models, no network)
pytest -m model                        # real-model checks (skipped when the env vars are unset)
```

The set is small and synthetic, so treat the numbers as a smoke check, not a benchmark. On real notes the ranking of modes reversed (fused beat dense; see the README). The not-in-notes threshold is weak on real notes: recalibrate it with `plm-kit calibrate` on your own questions.

`plm-kit eval --preset quality` applies the quality preset (top_n 20, weight 1.0) to the fused+rerank arm. The grid still sweeps top_n 6/10/20 x weight 0.25/0.5/1.0 as a measurement; it isn't a recommendation.

## Your own notes
`plm-kit eval --docs DIR --gold FILE.jsonl` runs the same report on your own notes and questions (same JSONL fields as
`golden.jsonl`; keep both outside the repo).

- `--fixed-threshold`: skip the 70/30 tuning and gate every query at your configured
  `retrieval.not_in_notes_threshold`. The report then has `threshold.fixed` and `threshold.full_gate`
  (caught / falsely gated over the whole set). Use it to measure the cutoff you actually run with.
- **Newest-copy step** (experimental, off by default): run once with `--newest-copy off` and once with
  `--newest-copy demote`, each with its own `--json` file. `--dedupe-threshold`, `--dedupe-multiplier` and
  `--dedupe-tie-tolerance` override the config values. The report's `info.newest_copy` shows how many groups and
  notes were found and how many groups have no clearly newest copy. `drop` can be tested but is not recommended.

## Future work
- A **live-over-archive** metric (does the current note rank above its stale or archived copies?). The golden format
  only lists relevant notes, not which copies are stale, so it needs a new label (e.g. `"stale": [...]`) first.
