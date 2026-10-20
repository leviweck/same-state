"""Content rule: nothing in the kit may name the maintainer, their employers (past or prospective), their
customers, or anyone from their private data. The kit ships synthetic data only.

The denylist is stored as SHA-256 hashes of normalized phrases (lowercase, runs of letters/digits, joined by one
space) so this file doesn't itself contain the names. Every 1-3 word window of every file in the repo (contents
and paths) is hashed and checked. Maintainers can add more names WITHOUT committing them: put one phrase per line
in a file outside the repo and set PLM_KIT_DENYLIST=/path/to/that/file, or in config/denylist.local.txt, which is
gitignored, never committed, never shipped, and picked up automatically when present (template:
config/denylist.example.txt). Run it with `python -m pytest -q tests/test_content_rules.py`."""
import hashlib
import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "node_modules", "dist", "build"}
MAX_WORDS = 3
LOCAL_DENYLIST = ROOT / "config" / "denylist.local.txt"   # gitignored; the list itself is never scanned
DENY = {
    "4378021eb3f94ab36397023a0ca9a347ae134769906ed3f686f3ecfb0e0a694f",
    "069a550a8f684000d913b2f7cd49d28b15a6443b052fae5907d0736eb252e927",
    "612ed4025a1035e8a7751e99f31929eedc71271d292780f9b773e0d4aa3d8a0f",
    "ce8cf2b15e7cac6d5ef34e9bd3efeaeca8abb776059ad805274debc3aa58e290",
    "0e06786ec3e8e324536146c4e10b80fe034c317c2eaa6eec4db7622dda67a20d",
    "7929bce4dc0187876525f9460132ac73e1de67d0e88dfed9045fffe34aa10eb6",
    "c2c7a302ad6cf2902a0d4c29572819181b9aecd1ae6ae2ced534eda2f35cf4db",
    "9c5fa3506188ba5fef20276dcec4a51c92cd53302ad098259b2862f34beeee88",
    "577fa4735c91637c65d0976ebd003882fe587bd7625767bd94a4826c2ca5dabb",
    "7b72a9521d7ce9157728e44ee03c62fceacfba645baa6ee9f6bf80174683906c",
    "c278ac1ee527d76908cf9787324872dc5f0b7baae20ee0875d86d01316f72f91",
    "c3c0a114bf9769f107003aca3157da439936a06b03e2d6f0e05965e742453bf2",
    "69654ee5cc49fbf2486330d139a096cac672d021e453928edc93b0d4a05638b6",
    "3a0376ecbf52163c46feade6c142d981649f2a4350d4d3486d6e8cf5eaf0c537",
    "329d027c880dfff51d2bec4fd808404ec9f2e08fa57d8385c8ab1546a5283264",
}

# Narrow, reviewed exceptions: (path, sha256 of the exact stripped line), approved by the maintainer: the NOTICE
# copyright line and the licensor/support-contact line (NOTICE, PRIVACY.md, README.md). Any other line in any file
# still fails.
ALLOWED_LINES = {
    ("NOTICE", "77e1e2bb3a95e88d1b5cdd25b2556c837ba59d9ff27a929c518e3ac4b7d1461d"),
    ("NOTICE", "b4d106598635c393e574a57a20e8d1257ddbc0d8d546870b54ae25ff7c0c959f"),
    ("PRIVACY.md", "b4d106598635c393e574a57a20e8d1257ddbc0d8d546870b54ae25ff7c0c959f"),
    ("README.md", "b4d106598635c393e574a57a20e8d1257ddbc0d8d546870b54ae25ff7c0c959f"),
}
# Local-only files that hold denied phrases on purpose (gitignored, never in the release allowlist).
SKIP_FILES = {"scripts/.release-denylist.local"}


def norm_tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def h(phrase):
    return hashlib.sha256(" ".join(norm_tokens(phrase)).encode()).hexdigest()


def deny_hashes():
    out = set(DENY)
    files = [pathlib.Path(os.environ["PLM_KIT_DENYLIST"]).expanduser()] if os.environ.get("PLM_KIT_DENYLIST") else []
    if LOCAL_DENYLIST.is_file():
        files.append(LOCAL_DENYLIST)
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                out.add(h(line))
    return out


def hits_in(text, deny):
    toks = norm_tokens(text)
    found = []
    for n in range(1, MAX_WORDS + 1):
        for i in range(len(toks) - n + 1):
            if hashlib.sha256(" ".join(toks[i:i + n]).encode()).hexdigest() in deny:
                found.append((i, n))
    return found


def repo_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = pathlib.Path(dirpath) / f
            if p != LOCAL_DENYLIST:
                yield p


def test_allowlist_is_exact():
    for path, line_hash in ALLOWED_LINES:
        lines = (ROOT / path).read_text().splitlines()
        assert any(hashlib.sha256(ln.strip().encode()).hexdigest() == line_hash for ln in lines), (
            f"stale ALLOWED_LINES entry for {path}: remove it")


def test_denylist_matcher_works():
    deny = {h("alpha beta"), h("gamma")}
    assert hits_in("x Alpha-Beta y", deny) and hits_in("GAMMA.", deny)
    assert not hits_in("alphabet gammas", deny)


def test_local_denylist_is_gitignored():
    import subprocess

    r = subprocess.run(["git", "check-ignore", "-q", str(LOCAL_DENYLIST)], check=False, cwd=ROOT,
                       capture_output=True)
    if r.returncode == 128:
        pytest.skip("not a git checkout")
    assert r.returncode == 0, "config/denylist.local.txt must be gitignored"


def test_no_denied_names_anywhere_in_the_repo():
    deny = deny_hashes()
    bad = []
    for p in repo_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in SKIP_FILES:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = [ln for ln in text.splitlines()
                 if (rel, hashlib.sha256(ln.strip().encode()).hexdigest()) not in ALLOWED_LINES]
        for i, n in hits_in(rel + "\n" + "\n".join(lines), deny):
            bad.append(f"{rel}: denied {n}-word phrase at token {i}")
    assert not bad, "\n".join(bad[:50])
