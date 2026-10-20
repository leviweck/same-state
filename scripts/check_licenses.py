"""License gate.
Fails on: any GPL/AGPL dependency; any unresolved UNKNOWN license; any weak-copyleft (MPL/LGPL) package that
Legal has not accepted; and any accepted package that appears vendored or patched inside the repo.
Usage: pip-licenses --from=mixed --format=json > licenses.json
       python scripts/check_licenses.py licenses.json [--repo .]"""
import argparse
import json
import pathlib
import re
import sys

FORBIDDEN = re.compile(r"(\bA?GPL\b|GNU (Affero )?General Public License)", re.IGNORECASE)
LESSER = re.compile(r"\bLGPL\b|Lesser", re.IGNORECASE)
WEAK = re.compile(r"(\bLGPL\b|Lesser General Public|\bMPL\b|Mozilla Public)", re.IGNORECASE)
# Packages whose metadata lacks a license field; verified manually (2026-10-17).
OVERRIDES = {
    "py-rust-stemmers": "MIT (bundled licenses/LICENSE in wheel; github.com/qdrant/py-rust-stemmers)",
    "sentencepiece": "Apache-2.0 (PyPI license_expression; github.com/google/sentencepiece)",
    "plm-kit": "Apache-2.0 (this kit itself, when installed with pip install -e .; see LICENSE)",
}
IGNORE = {"pip-licenses", "prettytable", "wcwidth", "pip", "setuptools", "wheel"}
POLICY = pathlib.Path(__file__).with_name("license_policy.json")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache"}
PATCH_EXT = (".patch", ".diff", ".orig", ".rej")


def vendoring_findings(repo, names):
    """Paths inside the repo that look like a vendored copy or a patch of an accepted package."""
    out = []
    repo = pathlib.Path(repo).resolve()
    for p in repo.rglob("*"):
        rel = p.relative_to(repo)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        low = p.name.lower()
        for n in names:
            if p.is_dir() and low in (n, n.replace("-", "_")):
                out.append(f"{rel}/ (directory named like accepted package '{n}')")
            elif p.is_file() and n in low and (low.endswith(PATCH_EXT) or low.endswith(".py")):
                out.append(f"{rel} (file named like accepted package '{n}')")
        if p.is_file() and low.endswith(PATCH_EXT):
            try:
                txt = p.read_text(errors="replace")
            except OSError:
                continue
            for n in names:
                if re.search(rf"(^|[/\s]){re.escape(n)}/", txt, re.MULTILINE):
                    out.append(f"{rel} (patch touches '{n}/')")
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("licenses_json")
    ap.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parents[1]))
    a = ap.parse_args(argv)
    with open(a.licenses_json) as f:
        rows = json.load(f)
    accepted = json.loads(POLICY.read_text())["accepted_weak_copyleft"]
    bad, unknown, weak_unaccepted, accepted_seen = [], [], [], []
    for r in rows:
        name, lic = r["Name"], r["License"]
        key = name.lower().replace("_", "-")
        if key in IGNORE:
            continue
        if lic.strip().upper() in ("UNKNOWN", "") and key in OVERRIDES:
            lic = OVERRIDES[key]
        if FORBIDDEN.search(lic) and not LESSER.search(lic):
            bad.append(f"{name}=={r['Version']}: {lic}")
        elif lic.strip().upper() in ("UNKNOWN", ""):
            unknown.append(f"{name}=={r['Version']}")
        elif WEAK.search(lic):
            if key in accepted:
                accepted_seen.append(f"{name}=={r['Version']}: {lic} [{accepted[key]['status']}; {accepted[key]['condition']}]")
            else:
                weak_unaccepted.append(f"{name}=={r['Version']}: {lic}")
    vend = vendoring_findings(a.repo, list(accepted))
    print(f"checked {len(rows)} packages")
    for x in accepted_seen:
        print("ACCEPTED (weak copyleft):", x)
    for x in weak_unaccepted:
        print("FAIL (weak copyleft not reviewed by Legal):", x)
    for x in unknown:
        print("FAIL (unknown license):", x)
    for x in bad:
        print("FAIL (GPL/AGPL):", x)
    for x in vend:
        print("FAIL (accepted MPL package appears vendored/patched; Legal acceptance void):", x)
    if bad or unknown or weak_unaccepted or vend:
        return 1
    print("OK: no GPL/AGPL; weak-copyleft packages accepted by Legal and not vendored/patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
