"""Voice lite (few-shot style, no training) and the verbatim-copy guard."""
import pytest

from plm_kit import cli, generate
from plm_kit import voice_lite as vl

EX = [
    "Quick answer: yes, that works, and I'd keep it simple so nobody has to think twice about it.",
    "Short version is the paper lands Monday, so anything after that slips a week, no drama.",
    "Honestly the cheapest fix is the boring one, so let's do that first and see if it holds.",
    "Good catch on the dates, I moved the call to Thursday afternoon when my head is clearer.",
    "Rule of thumb: under ten minutes, just do it.",
]


def write(tmp_path, lines, name="ex.txt"):
    f = tmp_path / name
    f.write_text("\n".join(lines) + "\n")
    return f


def test_load_examples_counts_and_formats(tmp_path):
    assert vl.load_examples(write(tmp_path, ["# comment", "", *EX])) == EX
    import json

    j = tmp_path / "ex.jsonl"
    j.write_text("".join(json.dumps({"text": e}) + "\n" for e in EX))
    assert vl.load_examples(j) == EX
    with pytest.raises(ValueError, match="5-8"):
        vl.load_examples(write(tmp_path, EX[:4]))
    with pytest.raises(ValueError, match="5-8"):
        vl.load_examples(write(tmp_path, EX + EX))
    with pytest.raises(ValueError, match="not found"):
        vl.load_examples(tmp_path / "missing.txt")


def test_flags_12_word_copy_but_not_11():
    words = EX[0].split()
    twelve = "So, " + " ".join(words[2:14]) + " - anyway."
    eleven = "So, " + " ".join(words[2:13]) + " - anyway."
    h = vl.find_copies(twelve, EX, 12)
    assert len(h) == 1 and h[0]["example"] == 0 and h[0]["words"] == 12
    assert vl.find_copies(eleven, EX, 12) == []


def test_copy_check_ignores_case_and_punctuation_and_reports_maximal_run():
    out = "QUICK ANSWER yes that works and i'd keep it simple so nobody has to think twice about it!!"
    h = vl.find_copies(out, EX, 12)
    assert len(h) == 1 and h[0]["words"] == len(vl._tokens(EX[0]))


def test_whole_short_example_is_flagged():
    assert vl.find_copies("My rule of thumb: under ten minutes, just do it. Then lunch.", EX, 12)[0]["example"] == 4
    assert vl.find_copies("Under ten minutes? Do it.", EX, 12) == []


def test_original_answer_passes():
    assert vl.find_copies("The deposit is half up front, then design starts.", EX, 12) == []


def test_prompt_marks_examples_as_style_only():
    m = vl.build_messages("Q?", EX)
    assert m[0]["role"] == "system" and "STYLE ONLY" in m[0]["content"] and EX[1] in m[0]["content"]
    assert m[1] == {"role": "user", "content": "Q?"}
    assert "new words" in vl.build_messages("Q?", EX, retry=True)[-1]["content"]


def test_guard_regenerates_then_redacts_or_blocks():
    copy = "Sure. " + EX[2] + " Done."
    replies = iter([copy, "A fresh answer in new words."])
    text, rep = vl.guarded_generate(lambda m, n: next(replies), "Q", EX, vl.settings({}), 50)
    assert text == "A fresh answer in new words." and rep["attempts"] == 2 and rep["action"] == "regenerated"

    text, rep = vl.guarded_generate(lambda m, n: copy, "Q", EX, vl.settings({}), 50)
    assert rep["action"] == "redacted" and vl.find_copies(text, EX, 12) == []
    assert text.startswith("Sure. ") and vl.REDACTED in text and "boring one" not in text

    st = vl.settings({"voice_lite": {"on_copy": "block", "retries": 0}})
    text, rep = vl.guarded_generate(lambda m, n: copy, "Q", EX, st, 50)
    assert rep["action"] == "blocked" and rep["attempts"] == 1 and "boring" not in text


def test_answer_local_voice_lite_never_returns_a_copy(tmp_path):
    f = write(tmp_path, EX)
    gen = {"backend": "ollama", "voice_lite": {"enabled": True, "examples_file": str(f), "retries": 0}}
    text, rep = generate.answer_local("Q", [("a.md", "x")], gen, chat=lambda m, n: EX[0] + " " + EX[1])
    assert vl.find_copies(text, EX, 12) == [] and rep["copy_guard"]["action"] == "redacted"
    text, rep = generate.answer_local("Q", [("a.md", "x")], gen, voice_lite=False, chat=lambda m, n: "plain")
    assert text == "plain" and "copy_guard" not in rep


def test_settings_validation():
    for bad in ({"copy_guard_ngram": 3}, {"copy_guard_ngram": 13}, {"on_copy": "warn"}, {"retries": -1},
                {"enabled": True}):
        with pytest.raises(ValueError):
            vl.settings({"voice_lite": bad})


def test_copy_check_cli_never_echoes_examples(tmp_path, capsys):
    f = write(tmp_path, EX)
    d = tmp_path / "draft.txt"
    d.write_text("Hi! " + EX[3])
    assert cli.main(["copy-check", str(d), "--examples", str(f)]) == 1
    out = capsys.readouterr().out
    assert "FLAG copied" in out and "example #4" in out and "Thursday" not in out
    d.write_text("Moved the call; Thursday works better.")
    assert cli.main(["copy-check", str(d), "--examples", str(f)]) == 0
