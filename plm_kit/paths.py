"""User cache location. Everything the user downloads or builds (models, ONNX exports, the index of their notes)
lives here, outside the repo."""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]


def cache_dir(override=None):
    p = pathlib.Path(override or os.environ.get("PLM_KIT_CACHE") or "~/.cache/plm-kit").expanduser().resolve()
    if p == REPO or REPO in p.parents:
        raise ValueError(f"refusing to use a cache inside the kit repo: {p}")
    return p


def onnx_dir(name, override=None):
    return cache_dir(override) / f"{name}-onnx"


def index_dir(cfg_value=None, override=None):
    if cfg_value:
        p = pathlib.Path(cfg_value).expanduser().resolve()
        if p == REPO or REPO in p.parents:
            raise ValueError(f"refusing to write the notes index inside the kit repo: {p}")
        return p
    return cache_dir(override) / "index"
