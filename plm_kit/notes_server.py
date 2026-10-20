"""Read-only notes server for agents (Model Context Protocol, stdio). Started with `plm-kit mcp`.

It lets an agent you run (Cursor, a Grok Bot running the kit on YOUR computer) recall from your notes and check a
draft. It is deliberately read-only:
- Tools: search_notes, get_note, draft_check, copy_check, voice_examples, status. Nothing else.
- No tool sends, posts, writes, deletes or opens a network connection. Sending stays with you: the agent shows you a
  draft and you approve it in your own app. A test fails if a send/write-like tool is ever added (READ_ONLY_TOOLS).
- stdio only: the server talks to the one program that started it on this computer. No port is opened.
- Results are bounded (integrations.mcp.max_k / max_note_chars) and only indexed notes can be read (no file paths).
- voice_examples returns nothing unless integrations.mcp.expose_voice_examples is true (your example lines would
  otherwise go to the agent's model provider).

Privacy: whatever a tool returns goes into the calling agent's context, and so to that agent's model provider. See
PRIVACY.md. The logic lives in NotesService (plain Python, testable without the MCP SDK); build_server() wraps it."""
import pathlib
import sys

READ_ONLY_TOOLS = ("search_notes", "get_note", "draft_check", "copy_check", "voice_examples", "status")
DEFAULTS = {"max_k": 10, "max_note_chars": 8000, "max_excerpt_chars": 1200, "expose_voice_examples": False}
INSTRUCTIONS = (
    "Read-only access to the owner's notes. Use search_notes to recall, get_note for more of one note, then "
    "draft_check and copy_check on any outbound draft. This server cannot send anything. Never send, post or "
    "submit on the owner's behalf without showing the final draft and getting the owner's explicit yes. If "
    "search_notes says in_notes is false or weak_match is true, say the answer may not be in the notes.")


def settings(cfg):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in ((cfg.get("integrations") or {}).get("mcp") or {}).items() if v is not None})
    for k in ("max_k", "max_note_chars", "max_excerpt_chars"):
        if int(s[k]) < 1:
            raise ValueError(f"integrations.mcp.{k} must be >= 1")
    return s


class NotesService:
    """The six read-only operations. `retriever_factory()` -> (Retriever, info); built lazily on first use."""

    def __init__(self, cfg, retriever_factory, words_file=None):
        self.cfg = cfg
        self.s = settings(cfg)
        self._factory = retriever_factory
        self._ret = None
        self._info = {}
        from plm_kit import paths

        self.words_file = pathlib.Path(words_file or paths.REPO / "config" / "safety_words.txt")

    def _retriever(self):
        if self._ret is None:
            self._ret, self._info = self._factory()
        return self._ret

    def _clip(self, text, n):
        return text if len(text) <= n else text[:n].rstrip() + " [...]"

    def search_notes(self, query, k=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        k = max(1, min(int(k), int(self.s["max_k"])))
        ret = self._retriever()
        res = ret.search(query, mode=self._info.get("mode"), top_k=k)
        out = {"in_notes": res.in_notes, "weak_match": bool(res.low_confidence),
               "best_dense": None if res.best_dense is None else round(float(res.best_dense), 4), "results": []}
        if not res.in_notes:
            out["message"] = "Not in your notes."
            return out
        seen = set()
        for cid in res.chunks:
            doc = ret.doc[cid]
            if doc in seen:
                continue
            seen.add(doc)
            out["results"].append({"rank": len(out["results"]) + 1, "note": doc,
                                   "excerpt": self._clip(ret.text[cid], int(self.s["max_excerpt_chars"]))})
            if len(out["results"]) >= k:
                break
        if res.low_confidence:
            out["message"] = "Weak match: this may not be in your notes."
        return out

    def get_note(self, note):
        """Text of one INDEXED note (by the id search_notes returned). Never a filesystem path lookup."""
        ret = self._retriever()
        cids = sorted(c for c, d in ret.doc.items() if d == note)
        if not cids:
            raise ValueError("unknown note id; use an id returned by search_notes")
        text, seen = [], set()
        for c in cids:
            t = ret.text[c]
            if t not in seen:
                seen.add(t)
                text.append(t)
        full = "\n\n".join(text)
        lim = int(self.s["max_note_chars"])
        return {"note": note, "chunks": len(cids), "truncated": len(full) > lim, "text": self._clip(full, lim)}

    def draft_check(self, text, flag_urls=False):
        from plm_kit import safety

        words = safety.load_words(self.words_file) if self.words_file.is_file() else []
        found = safety.check(text or "", words, allow_urls=not flag_urls)
        return {"flagged": bool(found), "findings": found,
                "note": "Advisory only. The owner still reads and approves every message before it is sent."}

    def _examples(self):
        from plm_kit import voice_lite as vl

        g = self.cfg.get("generation") or {}
        path = (g.get("voice_lite") or {}).get("examples_file")
        return vl.load_examples(path) if path else None

    def copy_check(self, text):
        """Flag runs copied word for word from the voice lite examples. Positions only, never example text."""
        from plm_kit import voice_lite as vl

        ex = self._examples()
        if ex is None:
            return {"checked": False, "reason": "no generation.voice_lite.examples_file configured", "flags": []}
        n = int(vl.settings(dict(self.cfg.get("generation") or {}, voice_lite=dict(
            (self.cfg.get("generation") or {}).get("voice_lite") or {}, enabled=False)))["copy_guard_ngram"])
        hits = vl.find_copies(text or "", ex, n)
        return {"checked": True, "ngram": n, "flagged": bool(hits),
                "flags": [{"example": h["example"] + 1, "words": h["words"], "start": h["start"], "end": h["end"]}
                          for h in hits]}

    def voice_examples(self):
        if not self.s["expose_voice_examples"]:
            return {"enabled": False, "examples": [],
                    "reason": "integrations.mcp.expose_voice_examples is false (the default)"}
        ex = self._examples()
        if ex is None:
            return {"enabled": False, "examples": [], "reason": "no generation.voice_lite.examples_file configured"}
        return {"enabled": True, "examples": ex,
                "rule": "Match the tone and rhythm; never copy an example word for word (run copy_check)."}

    def status(self):
        ret = self._retriever()
        r = self.cfg.get("retrieval") or {}
        return {"read_only": True, "tools": list(READ_ONLY_TOOLS), "notes": len(set(ret.doc.values())),
                "chunks": len(ret.doc), "mode": self._info.get("mode"), "dense": self._info.get("dense"),
                "gate": r.get("not_in_notes_gate", "block"), "threshold": r.get("not_in_notes_threshold"),
                "voice_examples_exposed": bool(self.s["expose_voice_examples"])}


def build_server(service):
    """Wrap a NotesService in an MCP server (the `mcp` Python SDK, MIT; optional: requirements-mcp.txt).
    Every tool is marked read-only."""
    import functools

    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import ToolAnnotations

    def clean_errors(fn):
        """Bad input (ValueError) reaches the agent as a clear tool error, not a server crash."""
        @functools.wraps(fn)
        def inner(*a, **k):
            try:
                return fn(*a, **k)
            except ValueError as e:
                raise ToolError(str(e)) from None
        return inner

    srv = MCPServer(name="plm-kit-notes", instructions=INSTRUCTIONS)
    ro = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

    @srv.tool(name="search_notes", annotations=ro,
              description="Search the owner's notes. Returns ranked excerpts with note ids, plus in_notes/weak_match.")
    @clean_errors
    def search_notes(query: str, k: int = 5) -> dict:
        return service.search_notes(query, k)

    @srv.tool(name="get_note", annotations=ro, description="Text of one indexed note, by an id from search_notes.")
    @clean_errors
    def get_note(note: str) -> dict:
        return service.get_note(note)

    @srv.tool(name="draft_check", annotations=ro,
              description="Flag emails, phone numbers, card-like numbers, token-like strings and listed words in a "
                          "draft. Advisory; the owner still approves before anything is sent.")
    @clean_errors
    def draft_check(text: str, flag_urls: bool = False) -> dict:
        return service.draft_check(text, flag_urls)

    @srv.tool(name="copy_check", annotations=ro,
              description="Flag wording copied verbatim from the owner's style examples (positions only).")
    @clean_errors
    def copy_check(text: str) -> dict:
        return service.copy_check(text)

    @srv.tool(name="voice_examples", annotations=ro,
              description="The owner's 5-8 style example lines, only if the owner turned exposure on.")
    @clean_errors
    def voice_examples() -> dict:
        return service.voice_examples()

    @srv.tool(name="status", annotations=ro, description="Index size, retrieval mode, gate and server settings.")
    @clean_errors
    def status() -> dict:
        return service.status()

    return srv


MCP_MISSING = ("`plm-kit mcp` needs the optional MCP Python SDK, which isn't installed in this Python environment. "
               "Install it with: pip install -r requirements-mcp.txt")


def mcp_available():
    import importlib.util

    try:
        return importlib.util.find_spec("mcp") is not None and importlib.util.find_spec("mcp.server") is not None
    except (ImportError, ValueError):
        return False


def serve(cfg, docs=None, cache=None):
    """`plm-kit mcp`: stdio server. Logs go to stderr only (stdout carries the protocol)."""
    from plm_kit import cli

    if not mcp_available():                          # checked first: no index is loaded without the SDK
        raise RuntimeError(MCP_MISSING)

    def factory():
        return cli.build_retriever(cfg, docs, cache, out=sys.stderr)

    try:
        srv = build_server(NotesService(cfg, factory))
    except ImportError as e:
        raise RuntimeError(MCP_MISSING) from e
    print("plm-kit notes server (read-only, stdio) ready", file=sys.stderr)
    srv.run("stdio")
    return 0
