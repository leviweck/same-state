"""Newest-copy step (off by default), gate modes, owner/derived list loading."""
import json
import os
import pathlib

import pytest
import yaml

from plm_kit import cli, config, dedupe, stale
from plm_kit.docs import load_chunks
from plm_kit.pipeline import Retriever, gate_mode
from plm_kit.store import note_mtimes
from tests.fakes import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
BODY = ("The guest network is called Lantern and the router sits in the hall closet. To reset it hold the "
        "button for ten seconds then wait for the blue light. Channel eleven works best upstairs. "
        + " ".join(f"detail{i} about setup step{i}." for i in range(80)) + " ")


def rcfg(**over):
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["retrieval"]
    c.update(over)
    return c


def notes(tmp_path, files):
    for i, (rel, text) in enumerate(files.items()):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        os.utime(p, (1_700_000_000 + i * 1000, 1_700_000_000 + i * 1000))   # later entries are newer
    return tmp_path


@pytest.fixture
def vault(tmp_path):
    return notes(tmp_path, {
        "wifi.md": "# Wi-Fi notes\n" + BODY,
        "copies/wifi.md": "# Wi-Fi notes (copy)\n" + BODY + "Added: the printer uses the 5 GHz band.",
        "other/wifi.md": "# Wi-Fi notes\nCompletely different text about a cafe hotspot password list " * 5,
        "bike.md": "# Bike\n" + "chain lube every two weeks and check tyre pressure " * 6,
    })


def test_defaults_off():
    d = rcfg()["dedupe"]
    assert dedupe.settings(rcfg())["newest_copy"] == "off" and d["threshold"] == 0.9
    r = Retriever(ROOT / "sample_data" / "docs", rcfg(), embedder=HashEmbedder())
    assert r.newest.groups == [] and r.newest.stats["groups"] == 0


def test_normalize_title_and_jaccard():
    assert dedupe.normalize_title("Wi-Fi notes (copy 2)") == "wi fi notes"
    assert dedupe.normalize_title("Plan v3 final") == "plan"
    assert dedupe.normalize_title("2024") == "2024"
    a = dedupe.shingles(BODY * 2)
    assert dedupe.jaccard(a, a) == 1.0 and dedupe.jaccard(a, dedupe.shingles("unrelated words " * 20)) == 0.0


def test_grouping_needs_title_and_content(vault):
    ch = load_chunks(vault)
    s = dedupe.settings({"dedupe": {"newest_copy": "demote"}})
    assert dedupe.find_groups(ch, s) == [["copies/wifi.md", "wifi.md"]]          # other/wifi.md: same title only
    assert dedupe.find_groups(ch, dict(s, threshold=1.0)) == []                    # not identical -> no group at 1.0
    short = load_chunks(notes(vault / "s", {"a.md": "# T\nshort", "b.md": "# T\nshort"}))
    assert dedupe.find_groups(short, s) == []                                      # under min_words


@pytest.mark.parametrize("mode", ["demote", "drop"])
def test_newest_kept_older_demoted_or_dropped(vault, mode):
    r = Retriever(vault, rcfg(dedupe={"newest_copy": mode}, per_note_cap=0), embedder=HashEmbedder(),
                  mtimes=note_mtimes(vault))
    assert r.newest.stats == {"mode": mode, "groups": 1, "notes_in_groups": 2, "largest_group": 2,
                              "groups_without_clear_newest": 0, "tie_tolerance_s": 60.0}
    res = r.search("guest network Lantern reset router", mode="fused")
    assert res.docs[0] == "copies/wifi.md"                                         # newest copy first
    if mode == "drop":
        assert "wifi.md" not in res.docs
    else:
        assert "wifi.md" in res.docs                                               # demoted, still there
    off = Retriever(vault, rcfg(per_note_cap=0), embedder=HashEmbedder(), mtimes=note_mtimes(vault))
    assert {"wifi.md", "copies/wifi.md"} <= set(off.search("guest network Lantern reset router", mode="fused").docs)


def test_newest_by_mtime_not_by_name(vault):
    os.utime(vault / "wifi.md", (1_800_000_000, 1_800_000_000))                    # now the root copy is newest
    r = Retriever(vault, rcfg(dedupe={"newest_copy": "drop"}, per_note_cap=0), embedder=HashEmbedder(),
                  mtimes=note_mtimes(vault))
    docs = r.search("guest network Lantern reset router", mode="fused").docs
    assert docs[0] == "wifi.md" and "copies/wifi.md" not in docs


def test_single_member_retrieved_is_untouched():
    nc = dedupe.NewestCopy([], {"dedupe": {"newest_copy": "drop"}})
    nc.groups, nc.group_of = [["a.md", "b.md"]], {"a.md": 0, "b.md": 0}
    assert nc.apply({1: 1.0, 2: 0.5}, {1: "a.md", 2: "c.md"}) == {1: 1.0, 2: 0.5}


def test_dedupe_config_validation(tmp_path):
    for bad in ({"newest_copy": "newest"}, {"threshold": 0}, {"multiplier": 2}):
        with pytest.raises(ValueError):
            dedupe.settings({"dedupe": bad})
    assert dedupe.settings({"dedupe": {"newest_copy": False}})["newest_copy"] == "off"   # YAML `off` -> False
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    c["retrieval"]["dedupe"]["newest_copy"] = "sometimes"
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):
        config.load_config(f)


def test_index_reports_groups_and_eval_flag(vault, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "cache"))
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    c["data"]["docs_dir"] = str(vault)
    c["retrieval"]["dedupe"]["newest_copy"] = "demote"
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(c))
    assert cli.main(["--config", str(f), "index"]) == 0
    assert json.loads(capsys.readouterr().out)["newest_copy"]["groups"] == 1
    gold = tmp_path / "g.jsonl"
    rows = [{"id": f"a{i}", "query": q, "relevant": [rel], "answerable": True} for i, (q, rel) in enumerate(
        [("guest network Lantern reset", "copies/wifi.md"), ("router hall closet blue light", "copies/wifi.md"),
         ("chain lube tyre pressure", "bike.md"), ("cafe hotspot password list", "other/wifi.md"),
         ("printer 5 GHz band", "copies/wifi.md")])]
    rows += [{"id": f"n{i}", "query": q, "relevant": [], "answerable": False}
             for i, q in enumerate(["volcano tax", "orbital mechanics", "sourdough hydration"])]
    gold.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for flag in ("off", "drop"):
        assert cli.main(["eval", "--no-grid", "--docs", str(vault), "--gold", str(gold), "--newest-copy", flag]) == 0
        rep = json.loads(capsys.readouterr().out)
        assert rep["info"]["newest_copy"]["mode"] == flag
        assert rep["info"]["newest_copy"]["groups"] == (1 if flag == "drop" else 0)


# ---------- gate modes ----------
def test_gate_modes():
    docs = ROOT / "sample_data" / "docs"
    q = "oil change hatchback"
    block = Retriever(docs, rcfg(not_in_notes_threshold=1.01), embedder=HashEmbedder()).search(q)
    assert block.in_notes is False and block.low_confidence
    warn = Retriever(docs, rcfg(not_in_notes_threshold=1.01, not_in_notes_gate="warn"), embedder=HashEmbedder()).search(q)
    assert warn.in_notes is True and warn.low_confidence and warn.docs
    off = Retriever(docs, rcfg(not_in_notes_threshold=1.01, not_in_notes_gate="off"), embedder=HashEmbedder()).search(q)
    assert off.in_notes is True and not off.low_confidence
    assert rcfg()["not_in_notes_gate"] == "block" and gate_mode({"not_in_notes_gate": False}) == "off"
    with pytest.raises(ValueError):
        gate_mode({"not_in_notes_gate": "soft"})


def test_cli_warn_gate_prints_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "_embedder", lambda cfg, cache: (HashEmbedder(), {"hf_id": "x/y", "revision": "a" * 40}, None))
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    c["retrieval"].update(not_in_notes_threshold=1.01, not_in_notes_gate="warn")
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(c))
    assert cli.main(["--config", str(f), "ask", "factory reset AX-7731"]) == 0
    out = capsys.readouterr()
    assert "home-wifi.md" in out.out and "weak match" in out.err


def test_calibrate_prints_trade_off(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "_embedder", lambda cfg, cache: (HashEmbedder(), {"hf_id": "x/y", "revision": "a" * 40}, None))
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in [{"query": "factory reset AX-7731", "answerable": True},
                                                   {"query": "sourdough starter feeding", "answerable": True},
                                                   {"query": "zzqx volcano tax filing", "answerable": False}]))
    assert cli.main(["calibrate", str(q)]) == 0
    rep = json.loads(capsys.readouterr().out)
    sug = rep["suggested"]
    assert {"caught", "false_blocks", "of_not_in_notes", "of_answerable"} <= set(sug)
    assert f"catches {sug['caught']} of 1" in rep["trade_off"] and "falsely blocks" in rep["trade_off"]
    assert "scores_overlap" in rep and "advice" in rep


# ---------- owner/derived list loading ----------
def test_lists_file_yaml_and_toml_and_precedence(tmp_path):
    s = stale.settings({"stale": {"lists_file": str(ROOT / "config" / "stale.example.yaml")}})
    assert stale.multiplier("my-notes/a.md", s) == pytest.approx(1.2)
    assert stale.multiplier("web-clips/a.md", s) == pytest.approx(0.6)
    assert stale.multiplier("my-notes/web-clips/a.md", s) == pytest.approx(1.2)          # owner beats derived
    assert stale.multiplier("old/web-clips/a.md", s) == pytest.approx(0.5 * 0.6)         # archive stacks
    assert stale.multiplier("archive/my-notes/a.md", s) == pytest.approx(0.5 * 1.2)
    assert stale.multiplier("tool/export/2026-01-31/a.md", s) == pytest.approx(0.6)      # regex
    t = tmp_path / "l.toml"
    t.write_text('owner_regex = ["^vault/core/"]\nderived_patterns = ["clips"]\n')
    s2 = stale.settings({"stale": {"lists_file": str(t), "derived_patterns": ["*-export"]}})
    assert stale.multiplier("vault/core/x.md", s2) == pytest.approx(1.2)
    assert stale.multiplier("clips/x.md", s2) == stale.multiplier("a-export/x.md", s2) == pytest.approx(0.6)


def test_lists_defaults_empty_and_validation(tmp_path):
    d = rcfg()["stale"]
    assert d["owner_patterns"] == [] and d["owner_regex"] == [] and d["derived_regex"] == [] and d["lists_file"] is None
    bad = tmp_path / "b.yaml"
    bad.write_text("owner_weight: 9\n")
    with pytest.raises(ValueError):
        stale.settings({"stale": {"lists_file": str(bad)}})
    with pytest.raises(ValueError):
        stale.settings({"stale": {"owner_regex": ["("]}})


def _nc(mtimes, **s):
    nc = dedupe.NewestCopy([], {"dedupe": {"newest_copy": "drop", **s}}, mtimes)
    nc.groups, nc.group_of = [["a.md", "b.md"]], {"a.md": 0, "b.md": 0}
    return nc


@pytest.mark.parametrize(("ta", "tb", "tol", "kept"), [
    (1000, 1000, 60, {"a.md", "b.md"}),      # exact tie -> no winner, no name tiebreak
    (1030, 1000, 60, {"a.md", "b.md"}),      # within tolerance (bulk sync) -> no winner
    (1060, 1000, 60, {"a.md", "b.md"}),      # exactly at tolerance -> still a tie
    (1061, 1000, 60, {"a.md"}),              # clearly newer -> a wins
    (1000, 1001, 0, {"b.md"}),               # tolerance 0: any difference wins
    (1000, 1000, 0, {"a.md", "b.md"}),       # tolerance 0: exact tie still no winner
    (None, 1000, 60, {"b.md"}),              # unknown mtime counts as oldest
    (None, None, 60, {"a.md", "b.md"}),      # no times at all -> no winner
])
def test_tie_tolerance(ta, tb, tol, kept):
    mt = {d: t for d, t in (("a.md", ta), ("b.md", tb)) if t is not None}
    nc = _nc(mt, tie_tolerance_s=tol)
    out = nc.apply({1: 1.0, 2: 0.9}, {1: "a.md", 2: "b.md"})
    assert {{1: "a.md", 2: "b.md"}[c] for c in out} == kept
    assert nc.stats["groups_without_clear_newest"] == (0 if len(kept) == 1 else 1)


def test_tied_vault_mtimes_skip_group(vault):
    for rel in ("wifi.md", "copies/wifi.md"):
        os.utime(vault / rel, (1_750_000_000, 1_750_000_000))
    r = Retriever(vault, rcfg(dedupe={"newest_copy": "drop"}, per_note_cap=0), embedder=HashEmbedder(),
                  mtimes=note_mtimes(vault))
    assert r.newest.stats["groups"] == 1 and r.newest.stats["groups_without_clear_newest"] == 1
    assert {"wifi.md", "copies/wifi.md"} <= set(r.search("guest network Lantern reset router", mode="fused").docs)
    with pytest.raises(ValueError):
        dedupe.settings({"dedupe": {"tie_tolerance_s": -1}})
    assert rcfg()["dedupe"]["tie_tolerance_s"] == 60


def test_eval_fixed_threshold(monkeypatch):
    from eval import run_eval

    def fake_build(cfg, docs=None):
        r = Retriever(run_eval.DOCS, cfg, embedder=HashEmbedder())
        return r, {"newest_copy": r.newest.stats}

    monkeypatch.setattr(run_eval, "build", fake_build)
    rep = run_eval.main(cfg=rcfg(not_in_notes_threshold=0.5), modes=["dense"], grid=False, fixed_threshold=True)
    assert rep["threshold"]["fixed"] == 0.5 and "tuned_on_calib" not in rep["threshold"]
    assert set(rep["threshold"]["full_gate"]) == {"nin_caught", "false_gated"}
    tuned = run_eval.main(cfg=rcfg(not_in_notes_threshold=0.5), modes=["dense"], grid=False)
    assert "tuned_on_calib" in tuned["threshold"]
    assert cli.parser().parse_args(["eval", "--fixed-threshold"]).fixed_threshold
