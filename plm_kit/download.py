"""User-triggered model download. Runs only when the user invokes `plm-kit download`.
Fetches pinned revisions from Hugging Face into the user cache (outside the repo) and, for the ONNX models,
runs the export locally on the user's machine. Nothing is bundled with the kit."""
import os
import subprocess
import sys

from plm_kit import paths
from plm_kit.models import DEFAULT_LOCAL_GENERATOR, GENERATORS, RERANKERS, local_generator, resolve_embedder

THIRD_PARTY = ("Third-party models are downloaded by you from Hugging Face and are licensed to you by their "
               "owners under their own terms, not by this kit.")


def plan(cfg, components):
    """[(component, name, spec)] for the requested components, resolved from config."""
    out = []
    if "embed" in components:
        name, spec = resolve_embedder(cfg["retrieval"]["dense"])
        out.append(("embed", name, spec))
    if "reranker" in components:
        name = cfg["retrieval"]["reranker"]["model"]
        out.append(("reranker", name, RERANKERS[name]))
    if "generator" in components:
        gen = cfg.get("generation", {})
        spec = dict(GENERATORS["qwen3-8b"])
        if gen.get("base_model") and gen.get("base_revision"):
            spec.update(hf_id=gen["base_model"], revision=gen["base_revision"])
        out.append(("generator", "qwen3-8b", spec))
    if "gguf" in components:   # no-GPU answer model for generation.backend: llamacpp_python / llamacpp_server
        name = ((cfg.get("generation") or {}).get("local") or {}).get("model") or DEFAULT_LOCAL_GENERATOR
        out.append(("gguf", name, local_generator(name)))
    return out


def license_notice(items, cache):
    lines = ["LICENSE NOTICE (read before downloading)", "", THIRD_PARTY, ""]
    for comp, _name, s in items:
        lines.append(f"- [{comp}] {s['hf_id']} @ {s['revision']}" + (f" (file {s['file']})" if s.get("file") else ""))
        lines.append(f"    license: {s['license']} ({s['license_url']})")
        lines.append(f"    model card: https://huggingface.co/{s['hf_id']}/blob/{s['revision']}/README.md")
        if s.get("size_in_gb"):
            lines.append(f"    approx. download: {s['size_in_gb']} GB")
    lines += ["", f"Destination (outside the kit repo): {cache}",
              "By continuing you accept each model's license from its owner."]
    return "\n".join(lines)


def fetch(spec, dest):
    """Download one pinned revision. Isolated so tests can mock it (no network in CI)."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download

    # weights + tokenizer/config only; never remote code (*.py)
    patterns = ["*.json", "*.safetensors", "*.txt"] if spec.get("task") is None else ["*.json", "model.safetensors", "*.txt"]
    return snapshot_download(spec["hf_id"], revision=spec["revision"], local_dir=str(dest), allow_patterns=patterns)


def fetch_file(spec, dest_dir):
    """Download ONE pinned GGUF file (no other files, never code) and check its sha256 against the registry."""
    import hashlib

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(spec["hf_id"], spec["file"], revision=spec["revision"], local_dir=str(dest_dir))
    if spec.get("sha256"):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != spec["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {spec['file']}: delete {p} and retry")
    return p


def export_onnx(src, task, out):
    """Local ONNX export on the user's machine (needs requirements-export.txt)."""
    try:
        subprocess.run(["optimum-cli", "export", "onnx", "--model", str(src), "--task", task, str(out)], check=True)
    except FileNotFoundError as e:
        raise RuntimeError("optimum-cli not found: install requirements-export.txt in this environment") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ONNX export failed (exit {e.returncode}); see the output above") from e


def run(cfg, components, cache=None, assume_yes=False, out=None, inp=input):
    out = out or sys.stdout
    cache = paths.cache_dir(cache)
    items = plan(cfg, components)
    print(license_notice(items, cache), file=out)
    if not assume_yes:
        try:
            ans = inp("Download now? Type 'yes' to continue: ").strip().lower()
        except (EOFError, KeyboardInterrupt):   # no input (piped/closed stdin) or Ctrl-C: treat as "no"
            print(file=out)
            ans = ""
        if ans != "yes":
            print("Cancelled. Nothing was downloaded.", file=out)
            return {}
    cache.mkdir(parents=True, exist_ok=True)
    done = {}
    for comp, name, spec in items:
        if comp == "gguf":
            done[comp] = str(fetch_file(spec, cache / "gguf"))
            print(f"ok: {comp} -> {done[comp]}", file=out)
            continue
        src = fetch(spec, cache / f"{name}-src")
        if comp in ("embed", "reranker"):
            dst = paths.onnx_dir(name, cache)
            export_onnx(src, spec["task"], dst)
            done[comp] = str(dst)
        else:
            done[comp] = str(src)
        print(f"ok: {comp} -> {done[comp]}", file=out)
    return done
