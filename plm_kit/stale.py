"""Stale-copy demotion: path-based score multipliers, applied per chunk AFTER RRF fusion (and in the rerank blend).
Nothing is dropped or deduplicated here; the per-note result cap lives in pipeline.py.

Per chunk, from its note's path (relative to the source root; matching is case-insensitive):
- archive: any FOLDER segment exactly in `archive_dirs`                      -> x archive_weight (default 0.5)
- owner:   any path segment (folder or file name) matches an `owner_patterns` glob -> x owner_weight (default 1.2)
- derived: any path segment matches a `derived_patterns` glob               -> x derived_weight (default 0.6)
owner and derived never combine: if both match, owner wins (an explicit list beats a generic guess).
Each of owner/derived can be given as globs (`*_patterns`, per path segment) and/or regexes (`*_regex`, searched in
the whole lowercase relative path), in config and/or an extra `lists_file` (config/stale.example.yaml).
archive stacks (multiplies) with either.

Known failure cases (documented in docs/RETRIEVAL.md): a live note inside a folder named e.g. `old` is demoted; a
stale copy renamed and stored elsewhere isn't caught; live and stale copies can both take result slots; stacked
weights can bury a note that is the only source."""
import fnmatch
import pathlib
import re

DEFAULT_ARCHIVE_DIRS = ("archive", "archived", "backup", "backups", "old")
# Conservative, generic: machine-made copies people commonly keep next to their notes.
DEFAULT_DERIVED = (
    "*-export", "*-exports", "*_export", "*_exports",      # folders like "notes-export", "chat_exports"
    "*.html.md", "*.html.txt",                            # HTML pages converted to text
    "*chat-summary*", "*chat-summaries*", "*chat-export*",  # exported chat summaries / transcripts
)
DEFAULTS = {"archive_dirs": list(DEFAULT_ARCHIVE_DIRS), "archive_weight": 0.5,
            "derived_patterns": list(DEFAULT_DERIVED), "derived_weight": 0.6,
            "owner_patterns": [], "owner_weight": 1.2,
            "owner_regex": [], "derived_regex": [],   # regexes, searched in the whole relative path
            "lists_file": None}                       # optional YAML/TOML with extra owner/derived lists
LIST_KEYS = ("owner_patterns", "owner_regex", "derived_patterns", "derived_regex")


def load_lists_file(path):
    """Read a user list file (see config/stale.example.yaml). Only the four list keys are accepted."""
    p = pathlib.Path(path).expanduser()
    if p.suffix.lower() == ".toml":
        import tomllib

        with p.open("rb") as f:
            data = tomllib.load(f)
    else:
        import yaml

        data = yaml.safe_load(p.read_text()) or {}
    bad = set(data) - set(LIST_KEYS)
    if bad:
        raise ValueError(f"{p.name}: unknown keys {sorted(bad)}; allowed {list(LIST_KEYS)}")
    for k, v in data.items():
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ValueError(f"{p.name}: {k} must be a list of strings")
    return data


def settings(cfg):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in ((cfg or {}).get("stale") or {}).items() if v is not None})
    if s.get("lists_file"):
        for k, v in load_lists_file(s["lists_file"]).items():   # extends the config lists (doesn't replace them)
            s[k] = list(s.get(k) or []) + v
    for k in ("owner_regex", "derived_regex"):
        try:
            s[f"_{k}"] = [re.compile(x, re.IGNORECASE) for x in s.get(k) or []]
        except re.error as e:
            raise ValueError(f"retrieval.stale.{k}: bad regex: {e}") from e
    for k in ("archive_weight", "derived_weight", "owner_weight"):
        if float(s[k]) < 0:
            raise ValueError(f"retrieval.stale.{k} must be >= 0")
    return s


def _match(segments, patterns):
    pats = [p.lower() for p in patterns]
    return any(fnmatch.fnmatchcase(seg, p) for seg in segments for p in pats)


def multiplier(doc_path, s):
    """Score multiplier for one note path, e.g. "archive/home-wifi-old.md"."""
    parts = [p.lower() for p in doc_path.replace("\\", "/").split("/") if p]
    folders = parts[:-1]
    m = 1.0
    if any(f in {d.lower() for d in s["archive_dirs"]} for f in folders):
        m *= float(s["archive_weight"])
    rel = "/".join(parts)
    if _match(parts, s["owner_patterns"]) or any(r.search(rel) for r in s.get("_owner_regex", [])):
        m *= float(s["owner_weight"])
    elif _match(parts, s["derived_patterns"]) or any(r.search(rel) for r in s.get("_derived_regex", [])):
        m *= float(s["derived_weight"])
    return m


def chunk_multipliers(chunks, cfg):
    """{cid: multiplier} for chunks whose multiplier isn't 1.0 (computed once per retriever, not per query)."""
    s = settings(cfg)
    by_doc, out = {}, {}
    for c in chunks:
        if c.doc not in by_doc:
            by_doc[c.doc] = multiplier(c.doc, s)
        if by_doc[c.doc] != 1.0:
            out[c.cid] = by_doc[c.doc]
    return out
