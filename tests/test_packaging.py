import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The packaging files stay with the maintainer; a release download has the tests but not these.
needs_packaging_files = pytest.mark.skipif(
    not (ROOT / "release-allowlist.txt").is_file() or not (ROOT / "scripts" / "package_release.sh").is_file(),
    reason="maintainer packaging files (release-allowlist.txt, scripts/package_release.sh) are not in a release download")


@needs_packaging_files
def test_allowlist_paths_exist_and_exclude_dev_files():
    lines = [ln.strip() for ln in (ROOT / "release-allowlist.txt").read_text().splitlines()]
    paths = [ln for ln in lines if ln and not ln.startswith("#")]
    for p in paths:
        assert (ROOT / p).exists(), p
    assert ".github" not in paths
    for private in ("docs/DECISIONS.md", "docs", "scripts", "scripts/.release-denylist.local", "config/denylist.local.txt",
                    "RELEASE_CHECKLIST.md", "release-allowlist.txt", "scripts/package_release.sh", "config"):
        assert private not in paths, private
    # shipped so the safety gates in AGENTS.md run in a download
    for gate in ("tests", "scripts/check_licenses.py", "requirements-dev.txt", ".gitleaks.toml", ".gitignore"):
        assert gate in paths, gate


@needs_packaging_files
def test_package_script_uses_git_archive_not_copy():
    s = (ROOT / "scripts" / "package_release.sh").read_text()
    assert "git archive" in s
    code = "\n".join(ln for ln in s.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"\b(cp -r|cp -a|rsync|zip -r)\b", code)


def test_version_matches_package():
    import tomllib

    import plm_kit

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["project"]["version"] == plm_kit.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", plm_kit.__version__)


@needs_packaging_files
def test_release_name_uses_version_and_commit():
    s = (ROOT / "scripts" / "package_release.sh").read_text()
    assert '"$REF^{commit}"' in s and 'NAME="plm-kit-v$VER-$SHA"' in s


def test_every_requirement_pinned_exactly():
    for f in ("requirements.txt", "requirements-dev.txt", "requirements-export.txt", "requirements.lock", "requirements-export.lock",
              "requirements-local.txt", "requirements-mcp.txt"):
        for ln in (ROOT / f).read_text().splitlines():
            ln = ln.split("#")[0].strip()
            if not ln or ln.startswith("-r"):
                continue
            assert re.match(r"^[A-Za-z0-9_.\-\[\]]+==[0-9][\w.\-]*$", ln), f"{f}: {ln}"


def test_serving_default_is_no_gpu_auto():
    import yaml

    g = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["generation"]
    assert g["backend"] == "auto"                       # no GPU needed out of the box; transformers_peft is opt-in
    assert re.fullmatch(r"[0-9a-f]{40}", g["base_revision"])


def test_package_discovery_is_explicit():
    """`pip install -e .` failed with 'Multiple top-level packages discovered in a flat-layout' (config/, eval/,
    starter_kit/ ... look like packages). Discovery must stay explicit and pick up plm_kit and its subpackages only."""
    import tomllib

    from setuptools import find_packages

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    include = cfg["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == ["plm_kit*"]
    found = find_packages(where=str(ROOT), include=include)
    assert "plm_kit" in found and "plm_kit.answer_backends" in found
    assert all(p == "plm_kit" or p.startswith("plm_kit.") for p in found)
    assert cfg["project"]["scripts"]["plm-kit"] == "plm_kit.cli:main"


def _package_into(tmp_path, args, out_env):
    """Run the working copy's package_release.sh against a throwaway local clone (no network), so the dev tree's
    real denylist, dist folder and uncommitted state are never touched."""
    import shutil
    import subprocess

    if not (ROOT / ".git").exists() or shutil.which("git") is None:
        pytest.skip("needs a git checkout")
    repo = tmp_path / "repo"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(ROOT), str(repo)], check=True)
    (repo / "scripts" / ".release-denylist.local").write_text("zz-no-such-internal-name-zz\n")
    env = {k: v for k, v in os.environ.items() if k != "OUT"}
    if out_env is not None:
        env["OUT"] = str(out_env)
    r = subprocess.run(["bash", str(ROOT / "scripts" / "package_release.sh"), *args], cwd=repo, env=env,
                       capture_output=True, text=True, check=False, timeout=300)
    assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
    return repo


@needs_packaging_files
def test_package_script_honors_out_env(tmp_path):
    out = tmp_path / "out"
    repo = _package_into(tmp_path, ["HEAD"], out)
    assert len(list(out.glob("plm-kit-v*.tar.gz"))) == 1 and len(list(out.glob("plm-kit-v*.tar.gz.sha256"))) == 1
    assert not (repo.parent / "plm-kit-dist").exists()


@needs_packaging_files
def test_package_script_positional_out_wins_over_env(tmp_path):
    pos, env_out = tmp_path / "pos", tmp_path / "env"
    _package_into(tmp_path, ["HEAD", str(pos)], env_out)
    assert len(list(pos.glob("plm-kit-v*.tar.gz"))) == 1 and not env_out.exists()
