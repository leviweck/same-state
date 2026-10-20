"""`plm-kit selftest`: the starter test kit. Confirms the kit works on YOUR machine in a few minutes, on CPU.

1. Indexes the bundled synthetic sample notes (starter_kit/sample_notes: a fictional one-person design studio, with
   archived and machine-exported copies) in memory. Your own index, cache and notes are not touched.
2. Runs the golden questions (starter_kit/golden_questions.jsonl), including questions the notes can't answer.
3. Prints a scorecard against starter_kit/thresholds.yaml and exits 0 (all PASS) or 1 (any FAIL).
   Without the embedding model it scores keyword search only (BM25 + demotion, keyword_only limits) and says so; with it, the default fused mode
   and the not-in-notes gate are scored too.
4. `--answer` also generates three answers with your no-GPU backend (generation.backend: ollama | llamacpp_server |
   llamacpp_python) and runs voice lite on the synthetic style examples, checking the verbatim-copy guard.
No network is used, except --answer talking to your local model server (loopback only by default)."""
import json
import pathlib
import sys
import time

import yaml

from plm_kit import paths
from plm_kit.metrics import hit_at_k, mrr, recall_at_k

KIT = paths.REPO / "starter_kit"
NOTES = KIT / "sample_notes"
GOLD = KIT / "golden_questions.jsonl"
THRESHOLDS = KIT / "thresholds.yaml"
STYLE_EXAMPLES = KIT / "style_examples.txt"
ANSWER_IDS = ("s01", "s07", "s17")


def load_gold(path=GOLD):
    return [json.loads(x) for x in pathlib.Path(path).read_text().splitlines() if x.strip()]


def _rank(docs, d):
    return docs.index(d) if d in docs else len(docs) + 1000


def score(runs, rows, threshold=None):
    """Scorecard numbers from {id: {"docs": [...], "best": float|None, "ms": float}}."""
    ans = [r for r in rows if r["answerable"]]
    nin = [r for r in rows if not r["answerable"]]
    m = {"answerable": len(ans), "not_in_notes": len(nin)}
    m["recall@3"] = round(sum(recall_at_k(runs[r["id"]]["docs"], r["relevant"], 3) for r in ans) / len(ans), 3)
    m["hit@1"] = round(sum(hit_at_k(runs[r["id"]]["docs"], r["relevant"], 1) for r in ans) / len(ans), 3)
    m["mrr"] = round(sum(mrr(runs[r["id"]]["docs"], r["relevant"]) for r in ans) / len(ans), 3)
    st = [r for r in ans if r.get("stale")]
    ok = 0
    for r in st:
        docs = runs[r["id"]]["docs"]
        best_live = min(_rank(docs, d) for d in r["relevant"])
        ok += best_live < len(docs) and all(best_live < _rank(docs, s) for s in r["stale"])
    m["stale_demoted"] = round(ok / len(st), 3) if st else None
    m["stale_questions"] = len(st)
    if threshold is not None and all(runs[r["id"]]["best"] is not None for r in rows):
        m["nin_caught"] = sum(runs[r["id"]]["best"] < threshold for r in nin)
        m["false_gated"] = sum(runs[r["id"]]["best"] < threshold for r in ans)
    lat = sorted(v["ms"] for v in runs.values())
    m["query_ms_p50"] = round(lat[len(lat) // 2], 1)
    return m


def check(m, limits):
    """[(name, value, rule, ok)] for every limit that applies to this arm."""
    out = []
    for k, v in limits.items():
        if k.startswith("min_"):
            key = k[4:]
            if m.get(key) is not None:
                out.append((key, m[key], f">= {v}", m[key] >= v))
        elif k.startswith("max_"):
            key = k[4:]
            if m.get(key) is not None:
                out.append((key, m[key], f"<= {v}", m[key] <= v))
    return out


def _embedder(cfg, cache):
    from plm_kit.cli import _embedder

    try:
        emb, _spec, warn = _embedder(cfg, cache)
    except (ValueError, OSError) as e:
        return None, str(e)
    return emb, warn


def run_answers(cfg, rows, backend=None, cache=None, chat=None, out=None):
    """Three plain answers + one voice-lite answer per question on the synthetic notes/examples.
    Gates: every answer non-empty, and no copied style example in any final output."""
    from plm_kit import generate
    from plm_kit import voice_lite as vl
    from plm_kit.docs import load_chunks

    out = out or sys.stdout
    chunks = load_chunks(NOTES)
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c.text)
    gen = dict(cfg.get("generation") or {})
    gen["voice_lite"] = dict(gen.get("voice_lite") or {}, examples_file=str(STYLE_EXAMPLES), enabled=False)
    examples = vl.load_examples(STYLE_EXAMPLES)
    n = int(vl.settings(gen)["copy_guard_ngram"])
    res = []
    for r in (x for x in rows if x["id"] in ANSWER_IDS):
        excerpts = [(d, "\n".join(by_doc[d])) for d in r["relevant"]]
        row = {"id": r["id"], "query": r["query"]}
        for label, voice in (("plain", False), ("voice_lite", True)):
            t0 = time.perf_counter()
            text, rep = generate.answer_local(r["query"], excerpts, gen, cache=cache, backend=backend,
                                              voice_lite=voice, chat=chat)
            row[label] = {"seconds": round(time.perf_counter() - t0, 1), "chars": len(text.strip()),
                          "has_expected": r.get("answer_contains", "").lower() in text.lower(),
                          "copies_in_output": len(vl.find_copies(text, examples, n)),
                          "guard": rep.get("copy_guard"), "text": text.strip()}
        res.append(row)
        print(f"  [{r['id']}] {r['query']}\n    plain ({row['plain']['seconds']} s): "
              f"{row['plain']['text'][:160]!r}\n    voice lite ({row['voice_lite']['seconds']} s): "
              f"{row['voice_lite']['text'][:160]!r}", file=out)
    ok_nonempty = all(x[k]["chars"] > 0 for x in res for k in ("plain", "voice_lite"))
    ok_nocopy = all(x["voice_lite"]["copies_in_output"] == 0 for x in res)
    expected = sum(x[k]["has_expected"] for x in res for k in ("plain", "voice_lite"))
    return {"rows": res, "checks": [("answers_non_empty", ok_nonempty, "all", ok_nonempty),
                                    ("style_copies_in_output", 0 if ok_nocopy else 1, "== 0", ok_nocopy)],
            "info": {"answers_with_expected_fact": f"{expected}/{2 * len(res)}"}}


def run(cfg, cache=None, answer=False, json_out=None, backend=None, embedder=None, out=None, chat=None):
    out = out or sys.stdout
    t_all = time.perf_counter()
    from plm_kit.pipeline import Retriever

    limits = yaml.safe_load(THRESHOLDS.read_text())
    rows = load_gold()
    rcfg = dict(cfg["retrieval"])
    rcfg["dedupe"] = dict(rcfg.get("dedupe") or {}, newest_copy="off")   # deterministic: no file-time step
    warn = None
    if embedder is None:
        embedder, warn = _embedder(cfg, cache)
    t0 = time.perf_counter()
    ret = Retriever(NOTES, rcfg, embedder=embedder)   # in memory: nothing is written to your cache or index
    index_s = round(time.perf_counter() - t0, 2)
    # Always the default fused path (RRF + stale-copy demotion). Without embeddings it fuses BM25 alone, exactly as
    # `plm-kit ask` does before `plm-kit download`, so the keyword-only limits apply.
    arm = "fused" if ret.dense is not None else "keyword_only"
    runs = {}
    for r in rows:
        t0 = time.perf_counter()
        res = ret.search(r["query"], mode="fused", top_k=5)
        runs[r["id"]] = {"docs": res.docs, "best": res.best_dense, "ms": (time.perf_counter() - t0) * 1e3}
    thr = rcfg.get("not_in_notes_threshold") if arm == "fused" else None
    m = score(runs, rows, thr)
    checks = check(m, limits[arm])
    report = {"arm": arm, "notes": len({c.doc for c in ret.chunks}), "chunks": len(ret.chunks),
              "questions": len(rows), "index_seconds": index_s, "threshold": thr, "metrics": m,
              "checks": [{"name": n, "value": v, "rule": rule, "pass": ok} for n, v, rule, ok in checks]}
    print("plm-kit selftest: synthetic sample notes (fictional), no network\n", file=out)
    if arm == "keyword_only":
        print("note: embedding model not found, so this scores KEYWORD search only (BM25 + stale-copy demotion). Run `plm-kit download "
              "--components embed` and re-run to score the default fused mode and the not-in-notes gate.\n"
              + (f"      ({warn})\n" if warn else ""), file=out)
    print(f"mode: {arm} | notes: {report['notes']} | chunks: {report['chunks']} | questions: {len(rows)} "
          f"({m['answerable']} answerable, {m['not_in_notes']} not in the notes) | index: {index_s} s | "
          f"median query: {m['query_ms_p50']} ms\n", file=out)
    print(f"{'check':<24}{'value':>10}  {'needs':<10}result", file=out)
    for n, v, rule, ok in checks:
        val = f"{v}/{m['not_in_notes'] if n == 'nin_caught' else m['answerable']}" if n in ("nin_caught", "false_gated") else v
        print(f"{n:<24}{val!s:>10}  {rule:<10}{'PASS' if ok else 'FAIL'}", file=out)
    all_ok = all(ok for *_, ok in checks)
    if "nin_caught" in m:
        print(f"\nnote: the not-in-notes gate (threshold {thr}) caught {m['nin_caught']} of {m['not_in_notes']} "
              "out-of-notes questions. A partial catch is expected: the gate is a trade-off, not a filter "
              "(README 'The not-in-notes gate').", file=out)
    if answer:
        print("\nanswers (no-GPU backend; the synthetic style examples drive voice lite):", file=out)
        try:
            a = run_answers(cfg, rows, backend=backend, cache=cache, chat=chat, out=out)
        except (RuntimeError, ValueError) as e:
            a = {"error": str(e), "checks": [("answers_generated", False, "ok", False)], "info": {}}
            print(f"  error: {e}", file=out)
        for n, v, rule, ok in a["checks"]:
            print(f"{n:<24}{v!s:>10}  {rule:<10}{'PASS' if ok else 'FAIL'}", file=out)
        for k, v in a.get("info", {}).items():
            print(f"{k:<24}{v!s:>10}  (info: tiny models often miss facts; this isn't gated)", file=out)
        report["answers"] = {k: v for k, v in a.items() if k != "checks"}
        report["answers"]["checks"] = [{"name": n, "value": v, "rule": rule, "pass": ok} for n, v, rule, ok in a["checks"]]
        all_ok = all_ok and all(ok for *_, ok in a["checks"])
    report["seconds_total"] = round(time.perf_counter() - t_all, 1)
    report["pass"] = all_ok
    print(f"\n{'PASS' if all_ok else 'FAIL'}: selftest finished in {report['seconds_total']} s", file=out)
    if json_out:
        pathlib.Path(json_out).write_text(json.dumps(report, indent=1))
    return 0 if all_ok else 1
