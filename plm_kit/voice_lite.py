"""Voice lite: answers in YOUR writing style from 5-8 example lines you supply (few-shot), with no training.

- The examples are style samples, never sources: the prompt says so, and they are never printed.
- Verbatim-copy guard: any run of `copy_guard_ngram` (default 12) or more consecutive words copied from an example,
  or a whole example of 4+ words, is flagged. The answer is regenerated (`retries`, default 1); if it still copies,
  the copied words are removed (`on_copy: redact`, default) or the answer is withheld (`on_copy: block`).
  Either way the kit never prints a copied example.

Examples file (generation.voice_lite.examples_file, keep it OUTSIDE the repo): one example per line in a .txt file
(blank lines and lines starting with # are ignored), or JSONL with {"text": "..."} per line."""
import json
import pathlib
import re

MIN_EXAMPLES, MAX_EXAMPLES = 5, 8
DEFAULTS = {"enabled": False, "examples_file": None, "copy_guard_ngram": 12, "on_copy": "redact", "retries": 1}
WHOLE_EXAMPLE_MIN_WORDS = 4      # a copied example this long is flagged even if shorter than the n-gram window
REDACTED = "[...]"
_WORD = re.compile(r"[\w']+", re.UNICODE)

STYLE_SYSTEM = (
    "You write answers in the user's own voice. Below are short samples of how the user writes. Match their tone, "
    "sentence length, punctuation and word choice. The samples are for STYLE ONLY: they are not facts and not "
    "sources, so never take facts from them, and never quote, copy or reuse their sentences or phrases. Write new "
    "sentences.\n\nStyle samples:\n{samples}")
RETRY_NOTE = ("Your previous draft reused wording from the style samples. Rewrite the answer in completely new "
              "words; do not reuse any phrase from the samples.")


def settings(gen_cfg):
    s = dict(DEFAULTS)
    s.update({k: v for k, v in ((gen_cfg or {}).get("voice_lite") or {}).items() if v is not None})
    n = int(s["copy_guard_ngram"])
    if not 4 <= n <= 12:
        raise ValueError("generation.voice_lite.copy_guard_ngram must be between 4 and 12 (12 = the documented "
                         "maximum; smaller is stricter)")
    if s["on_copy"] not in ("redact", "block"):
        raise ValueError("generation.voice_lite.on_copy must be 'redact' or 'block'")
    if int(s["retries"]) < 0:
        raise ValueError("generation.voice_lite.retries must be >= 0")
    if s.get("enabled") and not s.get("examples_file"):
        raise ValueError("generation.voice_lite.enabled needs generation.voice_lite.examples_file "
                         "(5-8 of your own lines, one per line, kept outside the repo)")
    return s


def load_examples(path):
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"voice lite examples file not found: {p}")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if p.suffix.lower() == ".jsonl":
            row = json.loads(line)
            line = str(row.get("text", "")).strip() if isinstance(row, dict) else ""
            if not line:
                raise ValueError(f"{p.name}: each JSONL line needs a non-empty \"text\"")
        out.append(line)
    if not MIN_EXAMPLES <= len(out) <= MAX_EXAMPLES:
        raise ValueError(f"{p.name}: voice lite needs {MIN_EXAMPLES}-{MAX_EXAMPLES} example lines, found {len(out)}")
    return out


def build_messages(user_prompt, examples, retry=False):
    samples = "\n".join(f"- {e}" for e in examples)
    msgs = [{"role": "system", "content": STYLE_SYSTEM.format(samples=samples)},
            {"role": "user", "content": user_prompt}]
    if retry:
        msgs.append({"role": "user", "content": RETRY_NOTE})
    return msgs


def _tokens(text):
    """[(lowercased word, start, end)] so flagged spans can be located in the original text."""
    return [(m.group().lower(), m.start(), m.end()) for m in _WORD.finditer(text or "")]


def find_copies(output, examples, n=12):
    """Flag text copied from the examples: every maximal run of >= n consecutive words that also appears in one
    example, plus any whole example (>= WHOLE_EXAMPLE_MIN_WORDS words) reproduced in the output.
    Returns [{"example": i, "words": k, "start": c0, "end": c1, "text": ...}] (character offsets into output)."""
    out_t = _tokens(output)
    words = [w for w, _, _ in out_t]
    hits = []
    for i, ex in enumerate(examples):
        ex_w = [w for w, _, _ in _tokens(ex)]
        win = min(n, len(ex_w)) if len(ex_w) >= WHOLE_EXAMPLE_MIN_WORDS else n
        if len(ex_w) < win or len(words) < win:
            continue
        grams = {tuple(ex_w[j:j + win]) for j in range(len(ex_w) - win + 1)}
        flagged = [tuple(words[j:j + win]) in grams for j in range(len(words) - win + 1)]
        j = 0
        while j < len(flagged):
            if not flagged[j]:
                j += 1
                continue
            k = j
            while k + 1 < len(flagged) and flagged[k + 1]:
                k += 1
            a, b = out_t[j][1], out_t[k + win - 1][2]
            hits.append({"example": i, "words": k + win - j, "start": a, "end": b, "text": output[a:b]})
            j = k + 1
    return sorted(hits, key=lambda h: (h["start"], -h["end"]))


def redact(output, hits):
    spans = []
    for h in sorted(hits, key=lambda h: h["start"]):
        if spans and h["start"] <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], h["end"])
        else:
            spans.append([h["start"], h["end"]])
    text = output
    for a, b in reversed(spans):
        text = text[:a] + REDACTED + text[b:]
    return text


def guarded_generate(chat, user_prompt, examples, s, max_tokens):
    """chat(messages, max_tokens) -> str. Returns (answer, report). The answer never contains a flagged copy."""
    n = int(s["copy_guard_ngram"])
    report = {"attempts": 0, "copies_flagged": [], "action": "none"}
    text, hits = "", []
    for attempt in range(int(s["retries"]) + 1):
        report["attempts"] = attempt + 1
        text = chat(build_messages(user_prompt, examples, retry=attempt > 0), max_tokens)
        hits = find_copies(text, examples, n)
        if not hits:
            report["action"] = "regenerated" if attempt else "none"
            return text, report
        report["copies_flagged"].append([{k: h[k] for k in ("example", "words")} for h in hits])
    if s["on_copy"] == "block":
        report["action"] = "blocked"
        return ("[voice lite withheld this answer: it kept copying your style examples word for word. Try again, "
                "or turn voice lite off for this question.]"), report
    report["action"] = "redacted"
    return redact(text, hits), report
