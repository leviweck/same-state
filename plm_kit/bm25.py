"""BM25 keyword search over chunks with SQLite FTS5 (bm25() ranking). Columns: note title, heading path, body."""
import re
import sqlite3

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def fts_query(text):
    """Turn free text into a safe FTS5 OR-query of quoted terms (no FTS syntax injection)."""
    terms = sorted({t.lower() for t in _TOKEN.findall(text) if len(t) > 1})
    return " OR ".join(f'"{t}"' for t in terms)


class BM25Index:
    def __init__(self, chunks, path=":memory:"):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(title, heading, body, "
                        "tokenize='porter unicode61')")
        self.db.execute("DELETE FROM chunks")
        self.db.executemany("INSERT INTO chunks(rowid, title, heading, body) VALUES (?, ?, ?, ?)",
                            [(c.cid, c.title, c.heading, c.body or c.text) for c in chunks])
        self.db.commit()

    def search(self, query, limit=50):
        """Return chunk ids, best first. bm25() is lower-is-better in SQLite."""
        q = fts_query(query)
        if not q:
            return []
        rows = self.db.execute(
            "SELECT rowid FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?", (q, limit)
        ).fetchall()
        return [r[0] for r in rows]
