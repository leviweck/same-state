import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_licenses.py"
OK_ROWS = [{"Name": "certifi", "Version": "1", "License": "Mozilla Public License 2.0 (MPL 2.0)"},
           {"Name": "tqdm", "Version": "1", "License": "MPL-2.0 AND MIT"},
           {"Name": "numpy", "Version": "1", "License": "BSD-3-Clause"}]


def run(tmp_path, rows, repo):
    f = tmp_path / "lic.json"
    f.write_text(json.dumps(rows))
    return subprocess.run([sys.executable, str(SCRIPT), str(f), "--repo", str(repo)], check=False, capture_output=True, text=True)


def test_accepted_mpl_passes_with_legal_status(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    r = run(tmp_path, OK_ROWS, repo)
    assert r.returncode == 0, r.stdout
    assert r.stdout.count("reviewed, accepted by Legal 2026-10-17") == 2


def test_the_kit_itself_is_not_unknown(tmp_path):
    r = run(tmp_path, [*OK_ROWS, {"Name": "plm-kit", "Version": "1", "License": "UNKNOWN"}], tmp_path)
    assert r.returncode == 0, r.stdout


def test_gpl_fails(tmp_path):
    r = run(tmp_path, [*OK_ROWS, {"Name": "x", "Version": "1", "License": "GNU General Public License v3 (GPLv3)"}], tmp_path)
    assert r.returncode == 1 and "GPL/AGPL" in r.stdout


def test_unaccepted_mpl_fails(tmp_path):
    r = run(tmp_path, [{"Name": "other", "Version": "1", "License": "MPL-2.0"}], tmp_path)
    assert r.returncode == 1 and "not reviewed" in r.stdout


def test_vendored_dir_refails(tmp_path):
    repo = tmp_path / "repo"
    (repo / "third_party" / "certifi").mkdir(parents=True)
    r = run(tmp_path, OK_ROWS, repo)
    assert r.returncode == 1 and "vendored/patched" in r.stdout


def test_patch_file_refails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "fix.patch").write_text("--- a/tqdm/std.py\n+++ b/tqdm/std.py\n")
    r = run(tmp_path, OK_ROWS, repo)
    assert r.returncode == 1 and "patch touches 'tqdm/'" in r.stdout


def test_repo_itself_is_clean():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_licenses

    assert check_licenses.vendoring_findings(ROOT, ["certifi", "tqdm"]) == []
