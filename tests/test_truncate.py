"""Embedding / rerank input caps (~512 tokens) and batched embedding into one matrix."""
import os
import pathlib

import numpy as np
import pytest

from plm_kit import rerank
from plm_kit.dense import embed_batched
from plm_kit.truncate import CHARS_PER_TOKEN, SPECIAL_RESERVE, Truncator


class WordTok:
    """Tokenizer stand-in: one token per whitespace-separated word, with character offsets."""

    def no_truncation(self):
        pass

    def no_padding(self):
        pass

    def encode(self, text, add_special_tokens=False):
        import re

        class E:
            pass

        e = E()
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        e.ids, e.offsets = list(range(len(spans))), spans
        return e


def test_tokenizer_cut_enforces_max_tokens():
    t = Truncator(512, tokenizer=WordTok())
    long = " ".join(f"w{i}" for i in range(5000))
    out = t.cut(long)
    assert t.count(out) == 512 - SPECIAL_RESERVE and long.startswith(out)
    assert t.cut("short text") == "short text"


def test_char_fallback_without_tokenizer(tmp_path):
    t = Truncator(512, tokenizer_dir=tmp_path)          # no tokenizer.json -> conservative char cap
    assert not t.uses_tokenizer
    assert len(t.cut("x" * 50000)) == (512 - SPECIAL_RESERVE) * CHARS_PER_TOKEN


def test_pair_cut_fits_budget():
    t = Truncator(512, tokenizer=WordTok())
    q, d = t.cut_pair(" ".join(["q"] * 1000), " ".join(["d"] * 5000))
    assert t.count(q) <= 128 and t.count(q) + t.count(d) + 2 * SPECIAL_RESERVE <= 512 + SPECIAL_RESERVE


def test_embed_batched_truncates_every_input_and_streams_batches():
    seen = []

    def fake_embed(batch):
        seen.append([len(x) for x in batch])
        for x in batch:
            yield np.full(4, float(len(x)), np.float32)

    t = Truncator(64, tokenizer=None)
    texts = ["a" * 10_000] * 70 + ["b"]
    m = embed_batched(fake_embed, texts, 4, t, batch_size=32)
    assert m.shape == (71, 4) and m.dtype == np.float32
    assert [len(b) for b in seen] == [32, 32, 7]
    assert max(max(b) for b in seen) <= (64 - SPECIAL_RESERVE) * CHARS_PER_TOKEN
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


def test_reranker_scorer_truncates_pairs():
    got = {}

    class Enc:
        def rerank(self, q, docs, batch_size):
            got["q"], got["docs"], got["bs"] = q, docs, batch_size
            return [0.0] * len(docs)

    t = Truncator(512, tokenizer=None)
    score = rerank.make_scorer(Enc(), t, batch_size=16)
    assert score("x" * 9000, ["y" * 9000, "short"]) == [0.0, 0.0]
    cap = 512 * CHARS_PER_TOKEN
    assert len(got["q"]) < cap and all(len(got["q"]) + len(d) <= cap for d in got["docs"])
    assert got["docs"][1] == "short" and got["bs"] == 16


EMB = os.environ.get("PLM_KIT_EMBED_ONNX")


@pytest.mark.skipif(not (EMB and (pathlib.Path(EMB) / "tokenizer.json").is_file()), reason="no local export")
def test_real_tokenizer_cut():
    pytest.importorskip("tokenizers")
    t = Truncator(512, tokenizer_dir=EMB)
    assert t.uses_tokenizer
    out = t.cut("lorem ipsum dolor " * 2000)
    assert t.count(out) <= 512 - SPECIAL_RESERVE
