# Reranker (optional, off by default)

- Default: `retrieval.mode: fused` (BM25 + dense fused with RRF, k=60); the reranker isn't loaded.
- Turn it on with the **quality preset**: `retrieval.preset: quality` or `plm-kit ask --preset quality` (= `fused+rerank`, `top_n: 20`, `weight: 1.0`, about 2 s/query on CPU). `retrieval.mode: fused+rerank` uses the `reranker` block, which defaults to the same values.
- Kill switch: `PLM_KIT_RERANKER=off` overrides config and the flag (falls back to fused).
- Allowed plug-in: `granite-r2` = `ibm-granite/granite-embedding-reranker-english-r2` @ `d09d3d6971b689bf9c23839e45a470874d46e13a` (Apache-2.0). The model card states: "we _do not use_ the popular MS-MARCO retrieval dataset in our training corpus due to its non-commercial license" ([card @ pinned rev](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2/blob/d09d3d6971b689bf9c23839e45a470874d46e13a/README.md)). That is the model owner's statement, not a claim by this kit. You download it; it is never bundled and is subject to its own license.
- ONNX: run on **your own machine**: `python scripts/export_onnx.py granite-r2 ~/.cache/plm-kit` (downloads the pinned revision from Hugging Face and exports locally), then set `retrieval.reranker.onnx_dir` to the printed path. The ModernBERT export takes only `input_ids` + `attention_mask` (no `token_type_ids`); fastembed 0.9.0 handles that.
- Disabled means inert: no fastembed import, no download, no model load (tested).
- Values: use the quality preset's `top_n: 20`, `weight: 1.0`. `threads: null` uses all CPU cores. Inputs are capped at `max_tokens: 512` per (question, note) pair.
- Results: on the kit's synthetic set the reranker did not beat dense-only. On real notes the picture reversed: in the maintainer's internal test (107 questions, CPU) fused scored MRR 0.647 / hit@1 0.505, dense 0.551 / 0.393, and the quality preset 0.695 / 0.551 at about 2 s per query. Better on both, but much slower on CPU, so it stays opt-in and off by default. `top_n 10` / `weight 0.25` is no longer suggested.
- Blend over top_n: `score = 1/(k + fused_rank) + weight/(k + ce_rank)`; each chunk is then multiplied by its stale-copy multiplier (`retrieval.stale`, see `docs/RETRIEVAL.md`).
- Fails closed: any load/score error leaves the fused order unchanged.
- Not allowed: models trained on MS MARCO (non-commercial data terms) or under non-commercial licenses (e.g. CC-BY-NC).
- Recommend enabling only if it is >= baseline on recall@k, hit@k, MRR with zero not-in-notes regressions on your eval set (`plm-kit eval --modes fused,fused+rerank`).
