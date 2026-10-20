"""Third-party usage reporting is off for every kit entry point (PRIVACY.md).

onnxruntime >= 1.29 (official builds) turns on Microsoft 1DS telemetry by default and writes a persistent device id to
~/.cache/Microsoft/DeveloperTools/.onnxruntime/ unless ORT_DISABLE_TELEMETRY=1 is set before it loads. huggingface_hub /
transformers report usage unless HF_HUB_DISABLE_TELEMETRY=1. plm_kit/__init__.py sets both; these tests fail if any
entry point can load one of those libraries before that happens."""
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = r'''
import importlib, importlib.abc, os, runpy, sys
WATCH = {"onnxruntime", "fastembed", "huggingface_hub", "transformers", "optimum"}
class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in WATCH and not (os.environ.get("ORT_DISABLE_TELEMETRY") == "1"
                                                and os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"):
            print("EARLY:" + name, flush=True)
            os._exit(3)
        return None
sys.meta_path.insert(0, Hook())
entry = sys.argv[1]
if entry.endswith(".py"):
    runpy.run_path(entry, run_name="not_main")
else:
    importlib.import_module(entry)
for heavy in ("onnxruntime", "fastembed"):   # what the kit's search path imports next (absent is fine)
    try:
        importlib.import_module(heavy)
    except ImportError:
        pass
print("OK", os.environ["ORT_DISABLE_TELEMETRY"], os.environ["HF_HUB_DISABLE_TELEMETRY"], flush=True)
'''
ENTRIES = ["plm_kit", "plm_kit.cli", "plm_kit.notes_server", "plm_kit.dense", "plm_kit.rerank", "plm_kit.download",
           "plm_kit.selftest", "plm_kit.pipeline", "eval.run_eval", "scripts/export_onnx.py"]


def _run_hook(entry, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in ("ORT_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY",
                                                            "DISABLE_TELEMETRY", "DO_NOT_TRACK")}
    env.update(extra_env or {})
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *filter(None, [env.get("PYTHONPATH")])])
    return subprocess.run([sys.executable, "-c", HOOK, entry], cwd=ROOT, env=env, capture_output=True, text=True,
                          check=False, timeout=120)


@pytest.mark.parametrize("entry", ENTRIES)
def test_no_telemetry_library_loads_before_the_switches_are_set(entry):
    r = _run_hook(str(ROOT / entry) if entry.endswith(".py") else entry)
    assert r.returncode == 0 and "OK 1 1" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_the_switches_win_over_a_shell_value():
    r = _run_hook("plm_kit", {"ORT_DISABLE_TELEMETRY": "0", "HF_HUB_DISABLE_TELEMETRY": "0"})
    assert r.returncode == 0 and "OK 1 1" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_the_hook_would_catch_an_early_import():
    # negative control: loading onnxruntime/fastembed without importing plm_kit first must be flagged
    r = _run_hook("json")
    assert r.returncode == 3 and "EARLY:onnxruntime" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_importing_plm_kit_sets_both():
    import plm_kit  # noqa: F401

    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1" and os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def _embed_model():
    p = os.environ.get("PLM_KIT_EMBED_ONNX")
    cand = pathlib.Path(p) if p else pathlib.Path(os.environ.get("PLM_KIT_CACHE") or "~/.cache/plm-kit").expanduser() \
        / "granite-embed-r2-onnx"
    return cand if (cand / "model.onnx").is_file() else None


@pytest.mark.skipif(importlib.util.find_spec("fastembed") is None or _embed_model() is None,
                    reason="needs fastembed and the exported embedding model (plm-kit download --components embed)")
def test_selftest_index_ask_leave_no_onnxruntime_device_id(tmp_path):
    home, tmp, cache = tmp_path / "home", tmp_path / "tmp", tmp_path / "cache"
    for d in (home, tmp, cache):
        d.mkdir()
    (cache / "granite-embed-r2-onnx").symlink_to(_embed_model(), target_is_directory=True)
    env = {k: v for k, v in os.environ.items() if k not in ("ORT_DISABLE_TELEMETRY", "XDG_CACHE_HOME", "PLM_KIT_CACHE")}
    env.update(HOME=str(home), TMPDIR=str(tmp), PLM_KIT_CACHE=str(cache),
               PYTHONPATH=os.pathsep.join([str(ROOT), *filter(None, [env.get("PYTHONPATH")])]))
    docs = str(ROOT / "sample_data" / "docs")
    run = "import sys; from plm_kit.cli import main; sys.exit(main(sys.argv[1:]))"
    for args in (["selftest"], ["index", "--docs", docs], ["ask", "how often do I lube the bike chain?", "--docs", docs]):
        r = subprocess.run([sys.executable, "-c", run, *args], cwd=tmp_path, env=env, capture_output=True, text=True,
                           check=False, timeout=600)
        assert r.returncode == 0, (args, r.stdout[-1500:], r.stderr[-1500:])
    assert not (home / ".cache" / "Microsoft").exists()
    assert not (home / ".ses").exists() and not (tmp / ".ses").exists()
    assert not list(home.rglob("deviceid"))
