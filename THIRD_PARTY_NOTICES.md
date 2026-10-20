# Third-party notices

This repository is licensed under Apache-2.0 (see `LICENSE` and `NOTICE`). The components below keep their own licenses. Model weights are **downloaded by the user from Hugging Face at the pinned revision and are never bundled**.

Third-party models are downloaded by you from Hugging Face and are licensed to you by their owners under their own terms, not by this kit.

| Component | Use | License | Source / pinned version | Notes |
|---|---|---|---|---|
| fastembed | ONNX embedding + cross-encoder runtime | Apache-2.0 ([text](https://github.com/qdrant/fastembed/blob/main/LICENSE)) | https://github.com/qdrant/fastembed, `==0.9.0` | |
| onnxruntime | Inference runtime (fastembed dependency) | MIT | https://github.com/microsoft/onnxruntime | |
| SQLite (FTS5) | BM25 keyword index | Public domain | https://sqlite.org/copyright.html | |
| NumPy | Dense vector math | BSD-3-Clause ([text](https://github.com/numpy/numpy/blob/main/LICENSE.txt)) | https://github.com/numpy/numpy, `==2.5.4` | Bundled sub-licenses (0BSD, MIT, Zlib, CC0-1.0) per its wheel metadata |
| PyYAML | Config loading | MIT | https://github.com/yaml/pyyaml | |
| tokenizers (Hugging Face) | Input-length caps (512 tokens) using the model's own tokenizer.json | Apache-2.0 | https://github.com/huggingface/tokenizers, `==0.21.4` | Already transitive via fastembed; now imported directly |
| MSAL for Python (`msal`) | **Optional** OneDrive sign-in (device code / interactive), token cache | MIT ([text](https://github.com/AzureAD/microsoft-authentication-library-for-python/blob/1.39.0/LICENSE)) | https://github.com/AzureAD/microsoft-authentication-library-for-python, `==1.39.0` | Imported only by `plm-kit onedrive ...`. Installed from PyPI, unmodified, not vendored |
| PyJWT | MSAL dependency | MIT | https://github.com/jpadilla/pyjwt, `==2.15.1` | Transitive (via msal) |
| cryptography | MSAL dependency | Apache-2.0 OR BSD-3-Clause | https://github.com/pyca/cryptography, `==50.0.2` | Transitive (via msal/PyJWT). Its wheel statically bundles OpenSSL (per its SBOM, OpenSSL 4.0.3, Apache-2.0) and Rust crates under MIT/Apache-2.0/BSD-3-Clause; one crate (`self_cell`) is dual "Apache-2.0 OR GPL-2.0-only", used under Apache-2.0 |
| cffi / pycparser | cryptography dependencies | MIT-0 / BSD-3-Clause | `==2.1.1` / `==3.11` | Transitive |
| requests | Read-only Graph HTTP client (GET only) | Apache-2.0 | https://github.com/psf/requests, `==2.34.2` | Already transitive via huggingface_hub; now also direct |
| MCP Python SDK (`mcp`, `mcp-types`) | **Optional** read-only notes server for agents (`plm-kit mcp`, stdio) | MIT (package metadata `License: MIT`; [text](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE)) | https://github.com/modelcontextprotocol/python-sdk, `mcp==2.3.0`, `mcp-types==2.3.0` | Optional: installed only with `requirements-mcp.txt` (not in `requirements.txt` / `requirements.lock`). Imported only by `plm-kit mcp`. Installed from PyPI, unmodified, not vendored |
| MCP SDK dependencies | Transitive (via `mcp`) | MIT: pydantic `2.14.1`, pydantic_core `2.50.1`, annotated-types `0.8.0`, typing-inspection `0.4.4`, anyio `4.15.1`, h11 `0.16.0`, jsonschema `4.26.0`, jsonschema-specifications `2025.9.1`, referencing `0.37.0`, rpds-py `2026.9.1`, attrs `26.1.0`, truststore `0.10.5`. BSD-3-Clause: starlette `1.9.0`, sse-starlette `3.5.0`, uvicorn `0.54.0`, httpx2 `2.15.1`, httpcore2 `2.15.1`, click `8.5.0`. Apache-2.0: opentelemetry-api `1.45.1`, python-multipart `0.0.32` | PyPI, versions in `requirements-mcp.txt` | Optional, installed only with `requirements-mcp.txt`. The kit uses stdio only; the SDK's HTTP server parts are installed but never started |
| xAI API (Grok) | **Optional** hosted answer backend (`generation.backend: xai`), off by default | Service terms, not a software license | https://x.ai/legal, https://docs.x.ai | No SDK: the kit calls the HTTPS API with the Python standard library. Used under the user's own xAI account and key |
| Qwen3-8B | Answer generation (user-downloaded) | Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/Qwen/Qwen3-8B @ `b968826d9c46dd6066d109eabc6255188de91218` | Served with transformers + peft |
| Qwen3-4B (GGUF `Qwen/Qwen3-4B-GGUF`; Ollama `qwen3:4b-instruct`) | **Optional** no-GPU answers, default (user-downloaded) | Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/Qwen/Qwen3-4B-GGUF @ `bc640142c66e1fdd12af0bd68f40445458f3869b`, file `Qwen3-4B-Q4_K_M.gguf` | Ollama tag is Ollama's build of Qwen3-4B-Instruct-2507 (same family/license); pulled by the user from Ollama |
| Granite 4.0 Micro (GGUF; Ollama `granite4:micro`) | **Optional** no-GPU answers (user-downloaded) | Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/ibm-granite/granite-4.0-micro-GGUF @ `ec48475f0c811d812fbfb61975717a9c36eeb652`, file `granite-4.0-micro-Q4_K_M.gguf` | |
| Qwen2.5-1.5B-Instruct (GGUF; Ollama `qwen2.5:1.5b`) | **Optional** low-RAM answers (user-downloaded) | Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF @ `91cad51170dc346986eccefdc2dd33a9da36ead9`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf` | The 3B and largest Qwen2.5 sizes are NOT Apache-2.0 and are not listed |
| Qwen2.5-0.5B-Instruct (GGUF; Ollama `qwen2.5:0.5b`) | **Optional** smoke-test answers (user-downloaded) | Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF @ `9217f5db79a29953eb74d5343926648285ec7e67`, file `qwen2.5-0.5b-instruct-q4_k_m.gguf` | |
| Ollama | **Optional** local model server the user installs (backend `ollama`) | MIT ([text](https://github.com/ollama/ollama/blob/main/LICENSE)) | https://github.com/ollama/ollama | Not bundled or installed by the kit; reached over HTTP on 127.0.0.1 |
| llama.cpp (`llama-server`) | **Optional** local model server the user installs (backend `llamacpp_server`) | MIT ([text](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE)) | https://github.com/ggml-org/llama.cpp | Not bundled; reached over HTTP on 127.0.0.1 |
| llama-cpp-python | **Optional** in-process backend (`llamacpp_python`, `requirements-local.txt`) | MIT | https://github.com/abetlen/llama-cpp-python, `==0.3.36` | Installed by the user from PyPI or its wheel index; unmodified, not vendored. The wheel bundles a compiled copy of llama.cpp / ggml (MIT, https://github.com/ggml-org/llama.cpp) |
| diskcache | Dependency of llama-cpp-python (`requirements-local.txt`, optional) | Apache-2.0 | https://github.com/grantjenks/python-diskcache, `==5.6.3` | Installed by pip; not vendored |
| Jinja2 | Dependency of llama-cpp-python (chat templates; optional) | BSD-3-Clause | https://github.com/pallets/jinja, `==3.1.6` | Installed by pip; not vendored |
| MarkupSafe | Dependency of Jinja2 (optional) | BSD-3-Clause | https://github.com/pallets/markupsafe, `==3.0.3` | Installed by pip; not vendored |
| NumPy | Dependency of llama-cpp-python (optional) | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | https://github.com/numpy/numpy, `==2.5.4` | Installed by pip; not vendored |
| typing_extensions | Dependency of llama-cpp-python (optional) | PSF-2.0 | https://github.com/python/typing_extensions, `==4.16.0` | Installed by pip; not vendored |
| transformers / peft / accelerate | Default serving stack | Apache-2.0 | `==4.55.4` / `==0.17.1` / `==1.10.1` | |
| bitsandbytes | 4-bit quantization | MIT | `==0.47.0` | |
| PyTorch | Tensor runtime | BSD-3-Clause | `==2.8.0` | |
| safetensors / sentencepiece / protobuf / huggingface_hub | Model loading | Apache-2.0 / Apache-2.0 / BSD-3-Clause / Apache-2.0 | `==0.6.2` / `==0.2.1` / `==6.32.1` / `==0.34.4` | |
| ibm-granite/granite-embedding-english-r2 | Dense retrieval embeddings (user-downloaded) | Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/ibm-granite/granite-embedding-english-r2 @ `47ea694b257b703fee9253d75c2b1f2985180498` | The model card states: "we _do not use_ the popular MS-MARCO retrieval dataset in our training corpus due to its non-commercial license" ([card @ pinned rev](https://huggingface.co/ibm-granite/granite-embedding-english-r2/blob/47ea694b257b703fee9253d75c2b1f2985180498/README.md)). This is the model owner's statement, not a claim by this kit. |
| ibm-granite/granite-embedding-reranker-english-r2 | **Optional** reranker, off by default (user-downloaded) | Apache-2.0 ([text](https://www.apache.org/licenses/LICENSE-2.0)) | https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2 @ `d09d3d6971b689bf9c23839e45a470874d46e13a` | The model card states: "we _do not use_ the popular MS-MARCO retrieval dataset in our training corpus due to its non-commercial license" ([card @ pinned rev](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2/blob/d09d3d6971b689bf9c23839e45a470874d46e13a/README.md)). This is the model owner's statement, not a claim by this kit. |

## Permissive transitive dependencies

Installed by pip as dependencies of the packages above, from PyPI, unmodified and not vendored. Versions are the pins in
`requirements.lock` (core) and `requirements-export.lock` (export-only). Licenses are taken from each package's own
metadata (`License-Expression`, `License` or classifiers) at that version.

| Package | Version | License | Lock | Pulled in |
|---|---|---|---|---|
| charset-normalizer | `3.5.2` | MIT | core, export | via requests |
| filelock | `3.32.3` | MIT | core, export | via huggingface_hub / torch |
| flatbuffers | `25.12.19` | Apache-2.0 | core, export | via onnxruntime |
| fsspec | `2026.7.0` | BSD-3-Clause | core, export | via huggingface_hub / torch |
| hf-xet | `1.7.0` | Apache-2.0 | core, export | via huggingface_hub |
| idna | `3.20` | BSD-3-Clause | core, export | via requests |
| loguru | `0.7.3` | MIT | core | via fastembed |
| mmh3 | `5.3.1` | MIT | core | via fastembed |
| mpmath | `1.3.0` | BSD (classifier: BSD License) | core, export | via sympy |
| networkx | `3.6.1` | BSD-3-Clause | core, export | via torch |
| packaging | `26.3` | Apache-2.0 OR BSD-2-Clause | core, export | via huggingface_hub / transformers |
| pillow | `12.3.0` | MIT-CMU | core | via fastembed |
| psutil | `7.2.2` | BSD-3-Clause | core | via accelerate / peft |
| py_rust_stemmers | `0.1.8` | MIT (license file in the wheel; metadata has no license field) | core | via fastembed |
| regex | `2026.9.29` | Apache-2.0 AND CNRI-Python | core, export | via transformers |
| setuptools | `78.1.0` | MIT | core, export | via torch |
| sympy | `1.14.0` | BSD (classifier: BSD License) | core, export | via torch |
| urllib3 | `2.8.0` | MIT | core, export | via requests |
| ml_dtypes | `0.6.0` | Apache-2.0 | export | via onnx |
| optimum-onnx | `0.1.0` | Apache-2.0 | export | via optimum[onnx] |

PyTorch bundles third-party components under other permissive licenses. `torch==2.8.0` (core) declares BSD-3-Clause;
its wheel's LICENSE lists the bundled components (MIT, BSD-3-Clause, BSD-2-Clause, Apache-2.0, Apache-2.0 with
exception, BSL-1.0, public domain). The export-only `torch==2.14.1` declares `Apache-2.0 AND Apache-2.0 WITH
LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`. Both are installed unmodified from the PyTorch
or PyPI index and are not vendored.

## Weak-copyleft (MPL-2.0) dependencies

Both are installed from PyPI as transitive dependencies. Neither is vendored, copied, or modified in this repo.

| Component | Use | License | Source / pinned version | Notes |
|---|---|---|---|---|
| certifi | CA bundle (transitive, via requests) | MPL-2.0 ([text](https://www.mozilla.org/en-US/MPL/2.0/)) | https://github.com/certifi/python-certifi, `==2026.7.22` | Used **unmodified**, installed from PyPI as a dependency; **not vendored** or copied into this repo. Source at the link above. **MPL-2.0 reviewed, accepted by Legal 2026-10-17** (unmodified, not vendored). `scripts/check_licenses.py` re-flags this if a copy or patch ever appears in the repo. |
| tqdm | Progress bars (transitive, via transformers / huggingface_hub) | MPL-2.0 AND MIT ([MPL-2.0 text](https://www.mozilla.org/en-US/MPL/2.0/), [project LICENCE](https://github.com/tqdm/tqdm/blob/master/LICENCE)) | https://github.com/tqdm/tqdm, `==4.70.1` | Used **unmodified**, installed from PyPI as a dependency; **not vendored** or copied into this repo. Source at the link above. **MPL-2.0 reviewed, accepted by Legal 2026-10-17** (unmodified, not vendored). `scripts/check_licenses.py` re-flags this if a copy or patch ever appears in the repo. |

## Other

Export-time only (user-run `plm-kit download` / `scripts/export_onnx.py`; `requirements-export.txt`, full lock `requirements-export.lock`; not shipped as dependencies): huggingface_hub (Apache-2.0), optimum (Apache-2.0), onnx (Apache-2.0), onnxruntime (MIT), transformers (Apache-2.0), PyTorch (BSD-3-Clause), PyYAML (MIT).

Other transitive dependencies: listed above, with exact pins in `requirements.lock` and `requirements-export.lock`. The license check (`scripts/check_licenses.py`) found no GPL/AGPL.
Dev only (`requirements-dev.txt`, not needed to use the kit): pip-licenses (MIT), ruff (MIT), pytest (MIT), pre-commit (MIT). gitleaks (MIT) is installed separately.

Explicitly NOT used: `Xenova/ms-marco-MiniLM-L-6-v2` / `cross-encoder/ms-marco-*` (its model card lists the MS MARCO dataset as training data, and MS MARCO's terms are non-commercial); `jinaai/jina-reranker-v2-*` and `jina-reranker-m0` (CC-BY-NC-4.0).
