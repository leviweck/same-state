"""Cap model input length so one long chunk or query can't blow up CPU memory.

The pinned embedding model's tokenizer allows 8192 tokens, and fastembed truncates at that limit. Attention memory
grows with the square of the sequence length, so we cap inputs at `max_tokens` (default 512) ourselves:
- with the model's tokenizer.json (from the local export): cut the text at the end of the last token that fits,
  leaving room for special tokens;
- without it: a conservative character cap (CHARS_PER_TOKEN chars per token)."""
import pathlib

DEFAULT_MAX_TOKENS = 512
SPECIAL_RESERVE = 4        # [CLS]/[SEP]-style tokens added by the model's post-processor (pairs add more)
CHARS_PER_TOKEN = 2        # fallback: well under typical English (~4 chars/token), so the cap errs short


class Truncator:
    def __init__(self, max_tokens=DEFAULT_MAX_TOKENS, tokenizer_dir=None, tokenizer=None):
        self.max_tokens = int(max_tokens)
        if self.max_tokens < SPECIAL_RESERVE + 8:
            raise ValueError("max_tokens is too small")
        self.tok = tokenizer
        if self.tok is None and tokenizer_dir is not None:
            path = pathlib.Path(tokenizer_dir) / "tokenizer.json"
            if path.is_file():
                try:
                    from tokenizers import Tokenizer

                    self.tok = Tokenizer.from_file(str(path))
                except (ImportError, OSError, ValueError):
                    self.tok = None
        if self.tok is not None:
            self.tok.no_truncation()
            self.tok.no_padding()

    @property
    def uses_tokenizer(self):
        return self.tok is not None

    def count(self, text):
        return len(self.tok.encode(text, add_special_tokens=False).ids) if self.tok else len(text) // CHARS_PER_TOKEN

    def cut(self, text, max_tokens=None):
        """Return the longest prefix of `text` within max_tokens (minus SPECIAL_RESERVE)."""
        n = (max_tokens or self.max_tokens) - SPECIAL_RESERVE
        if self.tok is None:
            cap = n * CHARS_PER_TOKEN
            return text if len(text) <= cap else text[:cap]
        enc = self.tok.encode(text, add_special_tokens=False)
        if len(enc.ids) <= n:
            return text
        return text[: enc.offsets[n - 1][1]]

    def cut_pair(self, query, doc):
        """For cross-encoder (query, doc) pairs: query gets at most a quarter of the budget, doc the rest."""
        q = self.cut(query, max(SPECIAL_RESERVE + 8, self.max_tokens // 4))
        room = self.max_tokens - self.count(q) - SPECIAL_RESERVE
        return q, self.cut(doc, max(SPECIAL_RESERVE + 8, room))


def cap_backend_tokenizer(model, max_tokens):
    """Belt and braces: also lower fastembed's own tokenizer truncation (8192 for the pinned embedder).
    Uses fastembed 0.9.0 internals (`.model.tokenizer`); silently skipped if they change."""
    tok = getattr(getattr(model, "model", None), "tokenizer", None)
    if tok is not None and hasattr(tok, "enable_truncation"):
        try:
            tok.enable_truncation(max_length=int(max_tokens))
            return True
        except (TypeError, ValueError):
            return False
    return False
