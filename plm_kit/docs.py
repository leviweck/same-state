"""Load Markdown/text notes and split them into heading-aware, windowed chunks.

- Front matter (a leading `---` ... `---` block) is stripped.
- The note is split at every Markdown heading (outside code fences); each section keeps its heading path.
- Each section is windowed at `size` chars with `overlap` chars of overlap. A window ends at the last newline in its
  back half, else the last space in its back half, else a hard cut at `size`. No chunk body exceeds `size`.
- Embedded text = "title > heading > subheading" + newline + chunk body. The keyword index covers title, heading
  path and body separately.
- Index-time skips: files over `max_file_mb`, skip-listed folders and file names (see DEFAULT_SKIP)."""
import fnmatch
import pathlib
import re
from dataclasses import dataclass

DEFAULT_SIZE = 800
DEFAULT_OVERLAP = 150
DEFAULT_MAX_FILE_MB = 2
# Folders never indexed: VCS, trash, editor/tool state. File names never indexed: backup/temp copies.
DEFAULT_SKIP = {
    "dirs": [".git", ".hg", ".svn", ".trash", "trash", ".trashed", "$recycle.bin", ".obsidian", ".logseq", ".vscode",
             ".idea", ".foam", ".dendron", "node_modules", "__pycache__"],
    "files": ["*.bak", "*.bak.*", "*~", ".#*", "*.swp", "*backup*"],
}
EXTS = (".md", ".markdown", ".txt")
_FRONT = re.compile(r"\A\ufeff?---[ \t]*\r?\n.*?\r?\n(---|\.\.\.)[ \t]*(\r?\n|\Z)", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_FM_TITLE = re.compile(r"^title:[ \t]*['\"]?(.+?)['\"]?[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    cid: int
    doc: str            # path relative to its source root (with the source prefix, e.g. "onedrive/...")
    text: str           # what gets embedded and shown: "title > heading path\nbody"
    archived: bool      # True if a folder segment is in the default archive list (informational; scoring uses
                        # plm_kit/stale.py with the configured lists)
    title: str = ""
    heading: str = ""   # "H2 > H3" path, "" for text before the first heading
    body: str = ""


def strip_front_matter(raw):
    m = _FRONT.match(raw)
    fm = m.group(0) if m else ""
    t = _FM_TITLE.search(fm) if fm else None
    return raw[m.end():] if m else raw, (t.group(1).strip() if t else None)


def sections(text):
    """[(heading_path list, section text)] split at every heading outside code fences.
    Returns (h1_title_or_None, sections)."""
    out, path, stack, buf, fence, h1 = [], [], [], [], None, None
    for line in text.splitlines():
        f = _FENCE.match(line)
        if f:
            fence = None if fence == f.group(1) else (fence or f.group(1))
        m = None if fence or f else _HEADING.match(line)
        if not m:
            buf.append(line)
            continue
        out.append((list(path), "\n".join(buf)))
        buf = []
        level, name = len(m.group(1)), m.group(2).strip()
        if level == 1 and h1 is None and not any(s.strip() for _, s in out):
            h1 = name                    # the note's own title heading: not repeated in the path
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, name))
        path = [n for _, n in stack]
    out.append((list(path), "\n".join(buf)))
    return h1, [(p, s.strip()) for p, s in out if s.strip()]


def window(text, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP):
    """Windows of at most `size` chars, `overlap` chars shared between neighbours. Cut at the last newline in the
    window's back half, else the last space there, else hard at `size`."""
    if size < 2 or not 0 <= overlap < size:
        raise ValueError("need size >= 2 and 0 <= overlap < size")
    text = text.strip()
    out, pos, n = [], 0, len(text)
    while pos < n:
        end = pos + size
        if end >= n:
            out.append(text[pos:].strip())
            break
        half = pos + size // 2
        cut = text.rfind("\n", half, end)
        if cut <= pos:
            cut = text.rfind(" ", half, end)
        if cut <= pos:
            cut = end
        piece = text[pos:cut].strip()
        if piece:
            out.append(piece)
        nxt = max(cut - overlap, pos + 1)
        if overlap and nxt > pos + 1:     # start the overlap on a word boundary when there is one
            sp = text.find(" ", nxt, cut)
            nxt = sp + 1 if sp != -1 and sp + 1 < cut else nxt
        while nxt < n and text[nxt].isspace():
            nxt += 1
        pos = nxt
    return [p for p in out if p]


def _skipped(rel_parts, skip):
    dirs = {d.lower() for d in skip.get("dirs", [])}
    if any(p.lower() in dirs for p in rel_parts[:-1]):
        return True
    name = rel_parts[-1].lower()
    return any(fnmatch.fnmatchcase(name, pat.lower()) for pat in skip.get("files", []))


def iter_note_files(root, skip=None, max_file_mb=DEFAULT_MAX_FILE_MB, exts=EXTS):
    """Yield (path, rel_posix) for indexable notes under root, sorted, skip list and size limit applied."""
    skip = DEFAULT_SKIP if skip is None else skip
    root = pathlib.Path(root)
    if not root.is_dir():
        return
    limit = float(max_file_mb) * 1024 * 1024 if max_file_mb else None
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in exts or not p.is_file():
            continue
        parts = p.relative_to(root).parts
        if _skipped(parts, skip) or (limit is not None and p.stat().st_size > limit):
            continue
        yield p, p.relative_to(root).as_posix()


def chunk_note(raw, fallback_title, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP):
    """[(title, heading_path_str, body, embed_text)] for one note."""
    body, fm_title = strip_front_matter(raw)
    h1, secs = sections(body)
    title = (fm_title or h1 or fallback_title).strip()
    out = []
    for path, text in secs:
        head = " > ".join(path)
        label = " > ".join([title, *path])
        for piece in window(text, size, overlap):
            out.append((title, head, piece, f"{label}\n{piece}"))
    return out or [(title, "", "", title)]


def load_chunks(root, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP, skip=None, max_file_mb=DEFAULT_MAX_FILE_MB):
    """Chunks for a folder, or for a list of (folder, doc_prefix) pairs (e.g. synced OneDrive files)."""
    if isinstance(root, (list, tuple)):
        chunks = []
        for folder, prefix in root:
            for c in load_chunks(folder, size, overlap, skip, max_file_mb):
                chunks.append(Chunk(len(chunks), f"{prefix}{c.doc}", c.text, c.archived, c.title, c.heading, c.body))
        return chunks
    from plm_kit.stale import DEFAULT_ARCHIVE_DIRS

    archive = {d.lower() for d in DEFAULT_ARCHIVE_DIRS}
    chunks = []
    for path, rel in iter_note_files(root, skip, max_file_mb):
        archived = any(part.lower() in archive for part in rel.split("/")[:-1])
        raw = path.read_text(encoding="utf-8", errors="replace")
        for title, head, body, text in chunk_note(raw, path.stem, size, overlap):
            chunks.append(Chunk(len(chunks), rel, text, archived, title, head, body))
    return chunks


def chunking_settings(data_cfg):
    """(size, overlap, skip, max_file_mb) from the `data` config block, with defaults."""
    c = (data_cfg or {}).get("chunking") or {}
    skip = (data_cfg or {}).get("skip")
    skip = DEFAULT_SKIP if skip is None else {"dirs": skip.get("dirs") or [], "files": skip.get("files") or []}
    return (int(c.get("size") or DEFAULT_SIZE), int(c.get("overlap") if c.get("overlap") is not None else DEFAULT_OVERLAP),
            skip, (data_cfg or {}).get("max_file_mb", DEFAULT_MAX_FILE_MB))
