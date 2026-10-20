"""Outbound draft check: flags personal data patterns and terms from an editable word list.
Advisory only. A human still reads and approves every message before it is sent."""
import pathlib
import re

PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"),
    "url": re.compile(r"\bhttps?://\S+"),
    "us_ssn_like": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "card_like": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "token_like": re.compile(r"\b(?:sk|pk|ghp|xox[abp]|AKIA)[-_A-Za-z0-9]{12,}\b"),
}


def _luhn(digits):
    s, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        s, alt = s + n, not alt
    return s % 10 == 0


def load_words(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [ln.strip().lower() for ln in p.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def check(text, words=(), allow_urls=True):
    """Return a list of findings: {"kind", "match"}. Empty list = nothing flagged."""
    out = []
    for kind, rx in PATTERNS.items():
        if kind == "url" and allow_urls:
            continue
        for m in rx.finditer(text):
            val = m.group(0)
            if kind == "card_like" and not _luhn(re.sub(r"\D", "", val)):
                continue
            out.append({"kind": kind, "match": val})
    low = text.lower()
    for w in words:
        if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low):
            out.append({"kind": "word_list", "match": w})
    return out
