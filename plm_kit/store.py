"""Persistent index of the user's notes, stored in the user cache (never in the repo).
Rebuilt automatically when the notes or the embedding model change."""
import hashlib
import json
import pathlib

import numpy as np

from plm_kit.docs import (
    DEFAULT_MAX_FILE_MB,
    DEFAULT_OVERLAP,
    DEFAULT_SIZE,
    DEFAULT_SKIP,
    Chunk,
    iter_note_files,
    load_chunks,
)


def note_mtimes(roots, chunking=None):
    """{doc: mtime} for every indexable note under the roots (a cheap stat walk; nothing is read)."""
    ch = chunking or default_chunking()
    roots = roots if isinstance(roots, (list, tuple)) else [(roots, "")]
    return {prefix + rel: p.stat().st_mtime for folder, prefix in roots
            for p, rel in iter_note_files(pathlib.Path(folder), ch["skip"], ch["max_file_mb"])}


def default_chunking():
    return {"size": DEFAULT_SIZE, "overlap": DEFAULT_OVERLAP, "skip": DEFAULT_SKIP, "max_file_mb": DEFAULT_MAX_FILE_MB}

VERSION = 3  # 3: heading-aware windowed chunks, title/heading/body columns (older indexes are rebuilt)


def docs_fingerprint(docs_dir, chunking=None):
    """docs_dir: a folder or a list of (folder, doc_prefix) pairs (see docs.load_chunks)."""
    ch = chunking or default_chunking()
    h = hashlib.sha256()
    roots = docs_dir if isinstance(docs_dir, (list, tuple)) else [(docs_dir, "")]
    for folder, prefix in roots:
        for p, rel in iter_note_files(pathlib.Path(folder), ch["skip"], ch["max_file_mb"]):
            h.update((prefix + rel).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def manifest_for(docs_dir, emb_spec, chunking=None):
    ch = chunking or default_chunking()
    emb = None if emb_spec is None else {k: emb_spec.get(k) for k in
                                         ("hf_id", "revision", "max_tokens", "query_prefix", "document_prefix")}
    return {"version": VERSION, "chunking": json.loads(json.dumps(ch, sort_keys=True)),
            "docs_sha256": docs_fingerprint(docs_dir, ch), "embedder": emb}


def save(index_dir, chunks, matrix, manifest):
    d = pathlib.Path(index_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks.json").write_text(json.dumps([c.__dict__ for c in chunks]))
    if matrix is not None:
        np.save(d / "embeddings.npy", matrix)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))


def load(index_dir, expected_manifest):
    """Return (chunks, matrix) if the stored index matches; else None."""
    d = pathlib.Path(index_dir)
    try:
        if json.loads((d / "manifest.json").read_text()) != expected_manifest:
            return None
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        mp = d / "embeddings.npy"
        matrix = np.load(mp) if expected_manifest["embedder"] and mp.exists() else None
        if expected_manifest["embedder"] and matrix is None:
            return None
        return chunks, matrix
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_or_load(docs_dir, index_dir, embedder, emb_spec, chunking=None):
    """Return (chunks, matrix, rebuilt: bool). Rebuilds when notes, chunking settings or the embedder change."""
    ch = chunking or default_chunking()
    man = manifest_for(docs_dir, emb_spec if embedder is not None else None, ch)
    got = load(index_dir, man)
    if got:
        return got[0], got[1], False
    chunks = load_chunks(docs_dir, ch["size"], ch["overlap"], ch["skip"], ch["max_file_mb"])
    matrix = embedder.embed([c.text for c in chunks]) if embedder is not None and chunks else None
    save(index_dir, chunks, matrix, man)
    return chunks, matrix, True
