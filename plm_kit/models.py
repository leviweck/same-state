"""Pinned model registry. Weights are never bundled: the user downloads them from Hugging Face
at the pinned revision and exports ONNX locally (see scripts/export_onnx.py), outside this repo."""

RERANKERS = {  # allow-list: Apache-2.0 / MIT only; training-data statements are the model owners' (see THIRD_PARTY_NOTICES.md)
    "granite-r2": {
        "hf_id": "ibm-granite/granite-embedding-reranker-english-r2",
        "revision": "d09d3d6971b689bf9c23839e45a470874d46e13a",
        "task": "text-classification",
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "size_in_gb": 0.6,
    },
}

EMBEDDERS = {
    "granite-embed-r2": {
        "hf_id": "ibm-granite/granite-embedding-english-r2",
        "revision": "47ea694b257b703fee9253d75c2b1f2985180498",
        "task": "feature-extraction",
        "pooling": "cls",          # per the model's 1_Pooling/config.json
        "normalize": True,
        "dim": 768,
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "size_in_gb": 0.6,
    },
}

GENERATORS = {
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "size_in_gb": 16.4,
    },
}

# No-GPU answer models (generation.backend: ollama | llamacpp_server | llamacpp_python). Apache-2.0 / MIT only.
# `file` + `revision` are what `plm-kit download --components gguf` fetches (first-party GGUF repos only);
# `ollama` is the Ollama library tag you `ollama pull` yourself. `ram_gb` is a rough estimate for the Q4 weights plus
# an 8k context (guidance, not measured on every machine). Only the 0.5B model was run end to end by the maintainer.
LOCAL_GENERATORS = {
    "qwen3-4b-instruct": {     # DEFAULT for 8-16 GB machines
        "hf_id": "Qwen/Qwen3-4B-GGUF",
        "revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
        "file": "Qwen3-4B-Q4_K_M.gguf",
        "sha256": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        "ollama": "qwen3:4b-instruct",   # Ollama's build of Qwen3-4B-Instruct-2507 (same family, Apache-2.0)
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "size_in_gb": 2.5,
        "ram_gb": 5,
        "note": "4B. Good answers on 8-16 GB laptops; the GGUF file is the hybrid Qwen3-4B with thinking switched off "
                "by the kit (no <think> output).",
    },
    "granite-4.0-micro": {
        "hf_id": "ibm-granite/granite-4.0-micro-GGUF",
        "revision": "ec48475f0c811d812fbfb61975717a9c36eeb652",
        "file": "granite-4.0-micro-Q4_K_M.gguf",
        "sha256": "97c417dcc0534b0737c74016fb2af083cb17c3b51eaac621192d23961b7024eb",
        "ollama": "granite4:micro",
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "size_in_gb": 2.1,
        "ram_gb": 4,
        "note": "3B alternative from a different publisher.",
    },
    "qwen2.5-1.5b-instruct": {  # 8 GB machines / older laptops
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "sha256": "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
        "ollama": "qwen2.5:1.5b",
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "size_in_gb": 1.1,
        "ram_gb": 3,
        "note": "Low-RAM option: faster, weaker answers.",
    },
    "qwen2.5-0.5b-instruct": {  # smoke tests only (the maintainer's end-to-end CPU check uses it)
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "revision": "9217f5db79a29953eb74d5343926648285ec7e67",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "sha256": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        "ollama": "qwen2.5:0.5b",
        "license": "apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "size_in_gb": 0.5,
        "ram_gb": 1.5,
        "note": "Tiny: proves the pipeline runs; answers are often wrong. Not for real use.",
    },
}
# NOT in the registry on purpose: 3B Qwen2.5 and the largest Qwen2.5 (non-Apache "Qwen research/other" licenses), and
# models under custom community licenses with use restrictions. Check any model you add yourself.
DEFAULT_LOCAL_GENERATOR = "qwen3-4b-instruct"

PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"}
_SHA = "0123456789abcdef"


def _validate(spec, kind):
    need = {"hf_id", "revision", "license", "license_url"} | ({"pooling", "dim", "normalize"} if kind == "embedder" else set())
    missing = need - set(spec)
    if missing:
        raise ValueError(f"{kind} spec missing {sorted(missing)}")
    if len(spec["revision"]) != 40 or any(c not in _SHA for c in spec["revision"]):
        raise ValueError(f"{kind} revision must be a full 40-char Hugging Face commit hash")
    if spec["license"].lower() not in PERMISSIVE:
        raise ValueError(f"{kind} license {spec['license']!r} is not in the permissive allow-list {sorted(PERMISSIVE)}")
    return spec


def resolve_embedder(dense_cfg):
    """Config `retrieval.dense.model` is a registry key (default 'granite-embed-r2'), or `retrieval.dense.custom`
    is a full spec dict (hf_id, revision, pooling cls|mean, dim, normalize, license, license_url) to swap models."""
    custom = dense_cfg.get("custom")
    if custom:
        spec = _validate({"task": "feature-extraction", "size_in_gb": 0.0, **custom}, "embedder")
        return spec.get("name", "custom-embedder"), spec
    name = dense_cfg.get("model", "granite-embed-r2")
    return name, _validate(EMBEDDERS[name], "embedder")


def local_generator(name):
    """Spec for a no-GPU answer model (registry key), validated like the other pinned models."""
    if name not in LOCAL_GENERATORS:
        raise ValueError(f"unknown generation.local.model {name!r}; choose from {sorted(LOCAL_GENERATORS)}")
    return _validate(LOCAL_GENERATORS[name], "local generator")
