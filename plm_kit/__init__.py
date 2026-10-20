"""plm-kit."""
import os

# Turn off third-party usage reporting for this process before any library that has it can load. Every kit entry
# point (CLI, MCP server, eval, scripts/export_onnx.py) imports this package first, and onnxruntime / fastembed /
# huggingface_hub / transformers are only imported inside plm_kit modules. Set unconditionally (not setdefault) so
# PRIVACY.md's "the kit sets ...=1" holds even if the shell had another value.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"      # Microsoft ONNX Runtime 1DS telemetry (on by default since 1.29)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"   # Hugging Face hub / transformers telemetry and user-agent details

__version__ = "1.0.4"
