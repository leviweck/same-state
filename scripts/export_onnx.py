"""User-run helper, executed on the user's own machine (same as `plm-kit download`):
download a pinned model from Hugging Face and export ONNX locally into a cache OUTSIDE the repo.
The kit never converts, hosts, or redistributes weights.
Usage: python scripts/export_onnx.py granite-r2 [cache]   |   python scripts/export_onnx.py granite-embed-r2 [cache]"""
import sys

from plm_kit import download, paths
from plm_kit.models import EMBEDDERS, RERANKERS


def main(name, cache=None):
    spec = {**RERANKERS, **EMBEDDERS}[name]
    cache = paths.cache_dir(cache)
    print(download.license_notice([("model", name, spec)], cache))
    src = download.fetch(spec, cache / f"{name}-src")
    out = paths.onnx_dir(name, cache)
    download.export_onnx(src, spec["task"], out)
    print(f"exported: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
