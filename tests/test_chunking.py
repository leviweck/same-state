"""Heading-aware windowed chunking, size limits, skip list."""
import itertools

import pytest

from plm_kit import docs
from plm_kit.bm25 import BM25Index
from plm_kit.docs import chunk_note, load_chunks, window


def test_20k_char_single_paragraph_never_exceeds_max_chars():
    for text in ("x" * 20000, "word " * 4000, ("A long sentence without breaks " * 700)):
        pieces = window(text, 800, 150)
        assert pieces and all(len(p) <= 800 for p in pieces)
    out = chunk_note("# Title\n" + "y" * 20000, "stem", size=800, overlap=150)
    assert len(out) >= 25 and all(len(body) <= 800 for _, _, body, _ in out)


def test_window_cuts_in_back_half_at_newline_then_space_then_hard():
    text = ("a" * 500 + "\n" + "b" * 200 + " " + "c" * 400)
    first = window(text, 800, 0)[0]
    assert first == "a" * 500                       # last newline in the back half wins over the later space
    text2 = "a" * 600 + " " + "b" * 600
    assert window(text2, 800, 0)[0] == "a" * 600     # no newline -> last space in the back half
    text3 = "a" * 100 + " " + "b" * 1500             # the only space is in the front half -> hard cut
    assert len(window(text3, 800, 0)[0]) == 800


def test_overlap_between_neighbours():
    words = [f"w{i:04d}" for i in range(600)]
    pieces = window(" ".join(words), 800, 150)
    assert len(pieces) > 2
    for a, b in itertools.pairwise(pieces):
        tail = a[-100:].split()[-1]
        assert tail in b[:200]                      # the next window starts inside the previous one's last ~150 chars
    assert window(" ".join(words), 800, 0)[1].split()[0] not in pieces[0]


def test_bad_window_settings():
    with pytest.raises(ValueError):
        window("abc", 100, 100)


def test_heading_path_frontmatter_and_fences():
    raw = ("---\ntitle: Garden plan\ntags: [x]\n---\n# Garden plan\nintro line\n## Beds\nbed text\n### North bed\n"
           "```\n# not a heading\n```\nnorth text\n## Tools\ntool text\n# Second top heading\ntop text\n")
    out = chunk_note(raw, "stem")
    heads = [h for _, h, _, _ in out]
    assert heads == ["", "Beds", "Beds > North bed", "Tools", "Second top heading"]
    assert all(t == "Garden plan" for t, _, _, _ in out)
    nb = out[2]
    assert "# not a heading" in nb[2] and nb[3].startswith("Garden plan > Beds > North bed\n")
    assert "tags:" not in "".join(b for _, _, b, _ in out)       # front matter stripped


def test_title_falls_back_to_file_stem_and_empty_note_still_indexed():
    assert chunk_note("just text", "my-note")[0][3] == "my-note\njust text"
    assert chunk_note("", "empty")[0][0] == "empty"


def test_keyword_index_covers_title_heading_and_body(tmp_path):
    (tmp_path / "n.md").write_text("# Zebra notes\n## Quokka section\nplain body words\n")
    ch = load_chunks(tmp_path)
    idx = BM25Index(ch)
    assert idx.search("zebra") and idx.search("quokka") and idx.search("plain")


def test_skip_list_and_size_limit(tmp_path):
    files = {"live.md": "live", "old-backup-notes.md": "x", "n.md.bak": "x", "n.bak.md": "x", "draft.md~": "x",
             ".git/x.md": "x", ".obsidian/w.md": "x", ".vscode/s.md": "x", ".trash/t.md": "x", "trash/t2.md": "x",
             "Backup/kept-but-demoted.md": "y", "big.md": "z" * (2 * 1024 * 1024 + 1)}
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    got = {c.doc for c in load_chunks(tmp_path)}
    assert got == {"live.md", "Backup/kept-but-demoted.md"}
    custom = {c.doc for c in load_chunks(tmp_path, skip={"dirs": [], "files": []}, max_file_mb=None)}
    assert {"big.md", ".git/x.md", "old-backup-notes.md"} <= custom


def test_chunking_settings_from_config():
    size, overlap, skip, mb = docs.chunking_settings({"chunking": {"size": 400, "overlap": 0}, "max_file_mb": 1,
                                                      "skip": {"dirs": ["x"]}})
    assert (size, overlap, mb) == (400, 0, 1) and skip == {"dirs": ["x"], "files": []}
    assert docs.chunking_settings({})[:2] == (800, 150)


def test_sample_docs_chunk_sizes():
    ch = load_chunks(docs.pathlib.Path(__file__).resolve().parents[1] / "sample_data" / "docs")
    assert ch and all(len(c.body) <= 800 for c in ch)
