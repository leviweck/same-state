"""Config loading. YAML (config/default.yaml) or TOML (see config.example.toml). In TOML, an empty string
means "not set" (TOML has no null), so "" is read as None."""
import os
import pathlib
import tomllib

import yaml

from plm_kit import paths

DEFAULT_CONFIG = paths.REPO / "config" / "default.yaml"
MODES = ("dense", "fused", "fused+rerank")   # user-selectable retrieval modes; "bm25" exists for eval only
DEFAULT_MODE = "fused"
# Named bundles of retrieval settings. Opt-in only; the default is no preset (mode fused, reranker unused).
PRESETS = {
    "quality": {"mode": "fused+rerank", "reranker": {"top_n": 20, "weight": 1.0}},  # ~2 s/query on CPU
}


def apply_preset(retrieval_cfg, name=None):
    """Return a copy of the retrieval config with preset `name` (or retrieval.preset) applied on top.
    A CLI --mode still overrides the preset's mode afterwards (see pipeline.resolve_mode)."""
    import copy

    name = name or retrieval_cfg.get("preset")
    out = copy.deepcopy(retrieval_cfg)
    if not name:
        return out
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    for k, v in PRESETS[name].items():
        out[k] = dict(out.get(k) or {}, **v) if isinstance(v, dict) else v
    out["preset"] = name
    return out


def _nullify(x):
    if isinstance(x, dict):
        return {k: _nullify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nullify(v) for v in x]
    return None if x == "" else x


def load_config(path=None):
    p = pathlib.Path(path or os.environ.get("PLM_KIT_CONFIG") or DEFAULT_CONFIG).expanduser()
    if p.suffix.lower() == ".toml":
        with p.open("rb") as f:
            cfg = _nullify(tomllib.load(f))
    else:
        cfg = yaml.safe_load(p.read_text())
    validate(cfg)
    return cfg


def validate(cfg):
    mode = cfg.get("retrieval", {}).get("mode", DEFAULT_MODE)
    if mode not in MODES:
        raise ValueError(f"retrieval.mode must be one of {list(MODES)}, got {mode!r}")
    preset = cfg.get("retrieval", {}).get("preset")
    if preset and preset not in PRESETS:
        raise ValueError(f"retrieval.preset must be one of {sorted(PRESETS)} or null, got {preset!r}")
    from plm_kit import dedupe, stale
    from plm_kit.pipeline import gate_mode

    r = cfg.get("retrieval", {})
    gate_mode(r)
    dedupe.settings(r)
    stale.settings(r)
    g = cfg.get("generation") or {}
    from plm_kit import answer, local_llm, voice_lite

    names = answer.available(g)
    if g.get("backend", answer.DEFAULT_BACKEND) not in names:
        raise ValueError(f"generation.backend must be one of {names}")
    local_llm.settings(g)
    from plm_kit.answer_backends import xai

    xai.settings(g.get("xai"))   # model/region combo, prices and cost caps (also checked when the backend starts)
    voice_lite.settings(g)
    if (g.get("local") or {}).get("model"):
        from plm_kit.models import local_generator

        local_generator(g["local"]["model"])
    od = cfg.get("sources", {}).get("onedrive") or {}
    if od.get("auth", "device_code") not in ("device_code", "interactive"):
        raise ValueError("sources.onedrive.auth must be 'device_code' or 'interactive'")
    return cfg
