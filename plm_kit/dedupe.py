"""Optional newest-copy step (retrieval.dedupe), OFF by default.

Grouping (once per retriever, from the indexed chunks): two notes are near-duplicates when their normalized titles
match AND the Jaccard similarity of their word-shingle sets is >= `threshold` (default 0.9, conservative).
Groups are the connected components of that relation.

At query time, when two or more notes of one group are among the fused candidates, the newest of those (by file
mtime) is kept as is and the others are multiplied by `multiplier` (mode "demote") or removed (mode "drop").

Risks (see docs/RETRIEVAL.md): mtime can mislead after a copy, restore or sync; a shared template can make
unrelated notes look alike; the newer note isn't always the live one."""
import re
from collections import defaultdict

MODES = ("off", "demote", "drop")
DEFAULTS = {"newest_copy": "off", "threshold": 0.9, "multiplier": 0.5, "shingle_words": 5, "min_words": 20,
            "tie_tolerance_s": 60}
_COPY_TOKENS = re.compile(r"^(copy|old|backup|bak|final|draft|new|latest|v\d+|\d+)$")
_WORD = re.compile(r"[a-z0-9]+")


def settings(cfg):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in ((cfg or {}).get("dedupe") or {}).items() if v is not None})
    if s["newest_copy"] in (False, None):
        s["newest_copy"] = "off"
    if s["newest_copy"] not in MODES:
        raise ValueError(f"retrieval.dedupe.newest_copy must be one of {list(MODES)}")
    if not 0 < float(s["threshold"]) <= 1 or not 0 <= float(s["multiplier"]) <= 1:
        raise ValueError("retrieval.dedupe: threshold must be in (0, 1], multiplier in [0, 1]")
    if float(s["tie_tolerance_s"]) < 0:
        raise ValueError("retrieval.dedupe.tie_tolerance_s must be >= 0")
    return s


def normalize_title(title):
    """Lowercase words with trailing copy markers dropped: "Wi-Fi notes (copy 2)" -> "wi fi notes"."""
    words = _WORD.findall(title.lower())
    while len(words) > 1 and _COPY_TOKENS.match(words[-1]):
        words.pop()
    return " ".join(words)


def shingles(text, n=5):
    w = _WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if w else set()


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def find_groups(chunks, s):
    """[[doc, ...], ...] groups of 2+ near-duplicate notes (sorted, deterministic)."""
    text, title = defaultdict(list), {}
    for c in chunks:
        text[c.doc].append(c.body or c.text)
        title.setdefault(c.doc, c.title or c.doc.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    by_title = defaultdict(list)
    for doc in sorted(text):
        by_title[normalize_title(title[doc])].append(doc)
    parent = {}

    def root(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for docs in by_title.values():
        if len(docs) < 2:
            continue
        sh = {}
        for d in docs:
            body = "\n".join(text[d])
            if len(_WORD.findall(body.lower())) >= int(s["min_words"]):
                sh[d] = shingles(body, int(s["shingle_words"]))
        cand = sorted(sh)
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                if jaccard(sh[a], sh[b]) >= float(s["threshold"]):
                    parent[root(b)] = root(a)
    comp = defaultdict(list)
    for d in parent:
        comp[root(d)].append(d)
    for r in list(comp):
        if r not in comp[r]:
            comp[r].append(r)
    return sorted(sorted(set(g)) for g in comp.values() if len(set(g)) > 1)


class NewestCopy:
    """Query-time helper. `mtimes` maps doc -> modification time (seconds); missing docs count as oldest.
    A group only has a winner when its newest retrieved member is more than `tie_tolerance_s` newer than the next
    one; otherwise (tied or near-tied times, e.g. after a bulk sync) the group is left alone. No name tiebreak."""

    def __init__(self, chunks, cfg, mtimes=None):
        self.s = settings(cfg)
        self.mode = self.s["newest_copy"]
        self.mtimes = mtimes or {}
        self.groups = find_groups(chunks, self.s) if self.mode != "off" else []
        self.group_of = {d: i for i, g in enumerate(self.groups) for d in g}

    @property
    def stats(self):
        return {"mode": self.mode, "groups": len(self.groups), "notes_in_groups": sum(map(len, self.groups)),
                "largest_group": max(map(len, self.groups), default=0),
                "groups_without_clear_newest": sum(self.winner(g) is None for g in self.groups),
                "tie_tolerance_s": float(self.s["tie_tolerance_s"])}

    def winner(self, members):
        """Newest member if it is clearly newer (> tie_tolerance_s) than every other member, else None."""
        t = sorted(((self.mtimes.get(d), d) for d in members), key=lambda x: -x[0] if x[0] is not None else float("inf"))
        if len(t) < 2 or t[0][0] is None:
            return None
        if t[1][0] is not None and t[0][0] - t[1][0] <= float(self.s["tie_tolerance_s"]):
            return None
        return t[0][1]

    def losers(self, docs):
        """Docs among `docs` that lose to a newer retrieved member of their group."""
        hit = defaultdict(list)
        for d in dict.fromkeys(docs):
            if d in self.group_of:
                hit[self.group_of[d]].append(d)
        out = set()
        for members in hit.values():
            newest = self.winner(members) if len(members) > 1 else None
            if newest is not None:
                out |= set(members) - {newest}
        return out

    def apply(self, scores, doc_of):
        """scores {cid: score} -> new dict with losers demoted (or removed in mode drop)."""
        if self.mode == "off" or not self.groups:
            return scores
        lose = self.losers(doc_of[c] for c in sorted(scores, key=lambda c: -scores[c]))
        if not lose:
            return scores
        if self.mode == "drop":
            return {c: v for c, v in scores.items() if doc_of[c] not in lose}
        m = float(self.s["multiplier"])
        return {c: v * m if doc_of[c] in lose else v for c, v in scores.items()}
