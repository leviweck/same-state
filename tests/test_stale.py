"""Stale-copy multipliers (after fusion, no drops), per-note cap, dense floor, quality preset."""
import pathlib

import pytest
import yaml

from plm_kit import config, stale
from plm_kit.docs import Chunk
from plm_kit.pipeline import Retriever, cap_per_note
from tests.fakes import HashEmbedder

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "sample_data" / "docs"
S = stale.settings({})


def rcfg(**over):
    c = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())["retrieval"]
    c.update(over)
    return c


@pytest.mark.parametrize("path,m", [
    ("notes/live.md", 1.0),
    ("archive/x.md", 0.5), ("Archived/x.md", 0.5), ("a/BACKUPS/x.md", 0.5), ("old/x.md", 0.5), ("backup/x.md", 0.5),
    ("older/x.md", 1.0), ("archive.md", 1.0),          # exact FOLDER segments only
    ("notes-export/x.md", 0.6), ("chat_exports/x.md", 0.6), ("page.html.md", 0.6), ("my-chat-summary.md", 0.6),
    ("archive/notes-export/x.md", 0.5 * 0.6),          # archive stacks with derived
])
def test_default_multipliers(path, m):
    assert stale.multiplier(path, S) == pytest.approx(m)


def test_owner_and_derived_never_combine_archive_stacks():
    s = stale.settings({"stale": {"owner_patterns": ["mine"]}})
    assert stale.multiplier("mine/x.md", s) == pytest.approx(1.2)
    assert stale.multiplier("mine/notes-export/x.md", s) == pytest.approx(1.2)     # owner wins, no 1.2*0.6
    assert stale.multiplier("old/mine/notes-export/x.md", s) == pytest.approx(0.5 * 1.2)
    assert stale.multiplier("old/notes-export/x.md", s) == pytest.approx(0.5 * 0.6)


def test_configurable_and_validated():
    s = stale.settings({"stale": {"archive_dirs": ["attic"], "archive_weight": 0.25, "derived_patterns": []}})
    assert stale.multiplier("attic/x.md", s) == 0.25 and stale.multiplier("archive/x.md", s) == 1.0
    assert stale.multiplier("notes-export/x.md", s) == 1.0
    with pytest.raises(ValueError):
        stale.settings({"stale": {"owner_weight": -1}})


def _chunks():
    return [Chunk(0, "live.md", "guest network password lantern", False, "live", "", "guest network password lantern"),
            Chunk(1, "archive/live-old.md", "guest network password lantern", True, "old", "", "guest network password lantern"),
            Chunk(2, "notes-export/live.md", "guest network password lantern", False, "x", "", "guest network password lantern"),
            Chunk(3, "other.md", "unrelated bicycle chain", False, "o", "", "unrelated bicycle chain")]


def test_multipliers_applied_after_fusion_without_dropping():
    r = Retriever(None, rcfg(per_note_cap=0), embedder=HashEmbedder(), chunks=_chunks())
    res = r.search("guest network password lantern", mode="fused")
    assert res.docs[:3] == ["live.md", "notes-export/live.md", "archive/live-old.md"]   # 1.0 > 0.6 > 0.5
    assert set(res.docs) >= {"live.md", "archive/live-old.md", "notes-export/live.md"}   # demoted, not dropped
    assert r.mult == {1: 0.5, 2: 0.6}


def test_multipliers_used_in_rerank_blend():
    r = Retriever(None, rcfg(per_note_cap=0), embedder=HashEmbedder(), chunks=_chunks(),
                  scorer=lambda q, d: [1.0] * len(d))
    assert r.search("guest network password lantern", mode="fused+rerank").docs[0] == "live.md"


def test_per_note_cap():
    assert cap_per_note([1, 2, 3, 4, 5], {1: "a", 2: "a", 3: "a", 4: "b", 5: "a"}, 2) == [1, 2, 4]
    ch = [Chunk(i, "s.md", f"S > Part {i}\nsourdough starter feeding ratio part {i}", False, "S", f"Part {i}",
                f"sourdough starter feeding ratio part {i}") for i in range(6)]
    for cap, n in ((2, 2), (1, 1), (0, 6)):
        r = Retriever(None, rcfg(per_note_cap=cap), embedder=HashEmbedder(), chunks=ch)
        assert len(r.search("sourdough starter feeding ratio", mode="fused").chunks) == n


def test_dense_floor_optional():
    r = Retriever(None, rcfg(per_note_cap=0), embedder=HashEmbedder(), chunks=_chunks())
    r.cfg = rcfg(per_note_cap=0, dense_floor=0.5)
    res = r.search("guest network password lantern", mode="fused")
    assert res.chunks and 3 not in res.chunks
    assert rcfg()["dense_floor"] is None


def test_archive_copy_still_below_live_copy_on_sample_docs():
    res = Retriever(DOCS, rcfg(), embedder=HashEmbedder()).search("guest network LanternGuest", mode="fused")
    assert res.docs.index("home-wifi.md") < res.docs.index("archive/home-wifi-old.md")


def test_quality_preset():
    base = rcfg()
    assert base["preset"] is None and base["mode"] == "fused"
    q = config.apply_preset(base, "quality")
    assert q["mode"] == "fused+rerank" and q["reranker"]["top_n"] == 20 and q["reranker"]["weight"] == 1.0
    assert base["mode"] == "fused"                          # not mutated
    assert config.apply_preset(dict(base, preset="quality"))["mode"] == "fused+rerank"
    with pytest.raises(ValueError):
        config.apply_preset(base, "fast")


def test_cli_preset_and_calibrate(tmp_path, monkeypatch, capsys):
    import json

    from plm_kit import cli, rerank

    monkeypatch.setenv("PLM_KIT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "_embedder", lambda cfg, cache: (HashEmbedder(), {"hf_id": "x/y", "revision": "a" * 40}, None))
    calls = []
    monkeypatch.setattr(rerank, "get_reranker", lambda c: calls.append((c["top_n"], c["weight"])) or (lambda q, d: [0.0] * len(d)))
    assert cli.main(["ask", "factory reset AX-7731", "--preset", "quality"]) == 0
    assert calls == [(20, 1.0)] and "preset quality" in capsys.readouterr().err
    assert cli.main(["ask", "factory reset AX-7731", "--preset", "quality", "--mode", "fused"]) == 0  # --mode wins
    assert len(calls) == 1
    q = tmp_path / "q.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in [{"query": "factory reset AX-7731", "answerable": True},
                                                     {"query": "sourdough starter feeding", "answerable": True},
                                                     {"query": "zzqx volcano tax filing", "answerable": False}]))
    capsys.readouterr()
    assert cli.main(["calibrate", str(q)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["questions"] == 3 and "threshold" in rep["suggested"] and rep["current"]["threshold"] == 0.78
    q.write_text(json.dumps({"query": "x", "answerable": True}))
    assert cli.main(["calibrate", str(q)]) == 2
