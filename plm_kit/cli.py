"""plm-kit command line.

  plm-kit download [--components embed,reranker,gguf] [--include-generator] [--cache DIR] [--yes]
  plm-kit index    [--docs DIR]
  plm-kit ask      "question" [--k 5] [--mode dense|fused|fused+rerank] [--preset quality] [--answer]
                   [--backend auto|ollama|llamacpp_server|llamacpp_python|transformers_peft] [--voice-lite|--no-voice-lite]
  plm-kit selftest [--answer] [--json OUT]   (starter test kit: sample notes + golden questions + scorecard)
  plm-kit copy-check FILE|- [--examples FILE] [--n 12]   (flag text copied word for word from your style examples)
  plm-kit calibrate QUESTIONS.jsonl   (pick a not-in-notes trade-off: catches vs false blocks; it can't fix overlap)
  plm-kit eval     [--k 5] [--modes bm25,dense,fused,fused+rerank] [--json OUT] [--no-grid] [--fixed-threshold]
  plm-kit health
  plm-kit draft-check FILE|-   [--words FILE]
  plm-kit onedrive login|sync|status|logout     (optional, read-only; docs/ONEDRIVE.md)
  plm-kit mcp      [--docs DIR]   (read-only notes server for agents over stdio; plm_kit/notes_server.py)

Retrieval mode: retrieval.mode in config (default fused); --mode overrides it for one command.
Nothing is downloaded except by `download` (models) and `onedrive sync` (your own files, read-only). Models,
ONNX exports, synced files and the notes index live in the user cache (PLM_KIT_CACHE, default ~/.cache/plm-kit),
outside the repo. See PRIVACY.md."""
import argparse
import json
import os
import pathlib
import sys
import time

from plm_kit import paths, rerank
from plm_kit.config import DEFAULT_CONFIG, MODES, PRESETS, load_config  # noqa: F401 - re-exported
from plm_kit.pipeline import resolve_mode


def _docs_dir(cfg, override):
    d = pathlib.Path(override or cfg["data"]["docs_dir"]).expanduser()
    return d if d.is_absolute() else paths.REPO / d


def _onedrive_dir(cfg, cache):
    """Local folder holding synced OneDrive files, if the source is enabled (no settings check, no network)."""
    od = (cfg.get("sources") or {}).get("onedrive") or {}
    if not od.get("enabled"):
        return None
    from plm_kit import onedrive

    return onedrive.sync_dir(od, cache)


def _roots(cfg, docs, cache):
    roots = [(_docs_dir(cfg, docs), "")]
    od = _onedrive_dir(cfg, cache)
    if od is not None and od.is_dir():
        roots.append((od, "onedrive/"))
    return roots


def _embedder(cfg, cache):
    from plm_kit.models import resolve_embedder

    name, spec = resolve_embedder(cfg["retrieval"]["dense"])
    onnx = cfg["retrieval"]["dense"].get("onnx_dir") or paths.onnx_dir(name, cache)
    if not cfg["retrieval"]["dense"].get("enabled", True):
        return None, None, None
    if not (pathlib.Path(onnx) / "model.onnx").is_file():
        return None, spec, f"dense model not found at {onnx}; run `plm-kit download` (BM25-only until then)"
    from plm_kit.dense import BATCH_SIZE, FastembedEmbedder
    from plm_kit.truncate import DEFAULT_MAX_TOKENS

    d = cfg["retrieval"]["dense"]
    mt = int(d.get("max_tokens") or DEFAULT_MAX_TOKENS)
    spec = dict(spec, max_tokens=mt)
    return FastembedEmbedder(spec, str(onnx), max_tokens=mt, batch_size=int(d.get("batch_size") or BATCH_SIZE)), spec, None


def _scorer(cfg, cache, mode):
    """Load the reranker only for mode fused+rerank; any other mode never touches it."""
    if mode != "fused+rerank":
        return None
    rcfg = dict(cfg["retrieval"]["reranker"], enabled=True)
    rcfg.setdefault("onnx_dir", None)
    rcfg["onnx_dir"] = rcfg["onnx_dir"] or str(paths.onnx_dir(rcfg["model"], cache))
    return rerank.get_reranker(rcfg)


def build_retriever(cfg, docs=None, cache=None, out=None, mode=None, preset=None):
    """mode: CLI override (None -> retrieval.mode). Returns (retriever, info) where info["mode"] is the
    effective mode. Raises RuntimeError if the mode needs a model that isn't available."""
    out = out or sys.stderr
    from plm_kit import store
    from plm_kit.config import apply_preset
    from plm_kit.pipeline import Retriever

    cfg = dict(cfg, retrieval=apply_preset(cfg["retrieval"], preset))
    mode, note = resolve_mode(cfg["retrieval"], mode)
    if note:
        print(f"note: {note}", file=out)
    if cfg["retrieval"].get("preset"):
        print(f"note: preset {cfg['retrieval']['preset']} (mode {mode}"
              + (", ~2 s/query on CPU)" if mode == "fused+rerank" else ")"), file=out)
    emb, spec, warn = _embedder(cfg, cache)
    if warn:
        if mode == "dense":
            raise RuntimeError(f"mode dense needs the embedding model: {warn.split(';')[0]}; run `plm-kit download`")
        print(f"note: {warn}", file=out)
    idx = paths.index_dir(cfg["data"].get("index_dir"), cache)
    roots = _roots(cfg, docs, cache)
    from plm_kit.docs import chunking_settings

    size, overlap, skip, max_mb = chunking_settings(cfg.get("data"))
    chunking = {"size": size, "overlap": overlap, "skip": skip, "max_file_mb": max_mb}
    chunks, matrix, rebuilt = store.build_or_load(roots, idx, emb, spec, chunking)
    scorer = _scorer(cfg, cache, mode)
    if mode == "fused+rerank" and scorer is None:
        print("note: reranker not available (run `plm-kit download --components reranker`); using fused", file=out)
        mode = "fused"
    mtimes = store.note_mtimes(roots, chunking) if (cfg["retrieval"].get("dedupe") or {}).get("newest_copy") not in (
        None, False, "off") else None
    ret = Retriever(None, cfg["retrieval"], embedder=emb, scorer=scorer, chunks=chunks, matrix=matrix, mtimes=mtimes)
    return ret, {"newest_copy": ret.newest.stats, "index_dir": str(idx), "rebuilt": rebuilt, "chunks": len(chunks), "dense": emb is not None,
                 "mode": mode, "preset": cfg["retrieval"].get("preset"), "sources": [p or "local" for _, p in roots]}


def cmd_download(a, cfg):
    from plm_kit import download

    comps = [c.strip() for c in a.components.split(",") if c.strip()]
    if a.include_generator:
        comps.append("generator")
    bad = set(comps) - {"embed", "reranker", "generator", "gguf"}
    if bad:
        print(f"unknown components: {sorted(bad)}", file=sys.stderr)
        return 2
    done = download.run(cfg, comps, cache=a.cache, assume_yes=a.yes)
    return 0 if done or not comps else 1


def cmd_index(a, cfg):
    t0 = time.perf_counter()
    _, info = build_retriever(cfg, a.docs, a.cache)
    info["seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(info, indent=1))
    return 0


def cmd_ask(a, cfg):
    ret, info = build_retriever(cfg, a.docs, a.cache, mode=a.mode, preset=a.preset)
    res = ret.search(a.question, mode=info["mode"], top_k=a.k)
    if not res.in_notes:
        print("Not in your notes." + (f" (best dense match {res.best_dense:.3f})" if res.best_dense is not None else ""))
        return 0
    if res.low_confidence:
        print(f"warning: weak match (best dense {res.best_dense:.3f} < {ret.cfg.get('not_in_notes_threshold')}); "
              "this may not be in your notes", file=sys.stderr)
    if res.best_dense is not None:
        print(f"(best dense match {res.best_dense:.3f}; gate {ret.cfg.get('not_in_notes_threshold')})", file=sys.stderr)
    excerpts, seen = [], set()
    for cid in res.chunks:
        if ret.doc[cid] in seen:
            continue
        seen.add(ret.doc[cid])
        excerpts.append((ret.doc[cid], ret.text[cid]))
    if a.answer:
        from plm_kit import generate

        try:
            text, rep = generate.answer_local(a.question, excerpts[:a.k], cfg["generation"], cache=a.cache,
                                              backend=a.backend, voice_lite=a.voice_lite)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        print(text)
        guard = rep.get("copy_guard") or {}
        if guard.get("copies_flagged"):
            print(f"copy-guard: copied wording from your style examples was flagged "
                  f"{len(guard['copies_flagged'])} time(s); action: {guard['action']}", file=sys.stderr)
        print("sources: " + ", ".join(d for d, _ in excerpts[:a.k]), file=sys.stderr)
        return 0
    for i, (doc, text) in enumerate(excerpts[:a.k], start=1):
        print(f"[{i}] {doc}\n{text}\n")
    return 0


def cmd_eval(a, cfg):
    from eval import run_eval

    modes = [m.strip() for m in a.modes.split(",") if m.strip()] if a.modes else None
    bad = set(modes or []) - set(run_eval.ARMS)
    if bad:
        print(f"unknown eval modes: {sorted(bad)} (choose from {list(run_eval.ARMS)})", file=sys.stderr)
        return 2
    from plm_kit.config import apply_preset

    rcfg = apply_preset(cfg["retrieval"], a.preset)
    over = {k: v for k, v in (("newest_copy", a.newest_copy), ("threshold", a.dedupe_threshold),
                              ("multiplier", a.dedupe_multiplier), ("tie_tolerance_s", a.dedupe_tie_tolerance))
            if v is not None}
    if over:
        d = dict(rcfg.get("dedupe") or {})
        d.update(over)
        rcfg = dict(rcfg, dedupe=d)
    run_eval.main(a.k, a.json, grid=not a.no_grid, modes=modes, cfg=rcfg, docs=a.docs, gold=a.gold,
                  fixed_threshold=a.fixed_threshold)
    return 0


def cmd_selftest(a, cfg):
    from plm_kit import selftest

    return selftest.run(cfg, cache=a.cache, answer=a.answer, json_out=a.json, backend=a.backend)


def cmd_copy_check(a, cfg):
    from plm_kit import voice_lite as vl

    vs = vl.settings(dict(cfg.get("generation") or {}, voice_lite=dict(
        (cfg.get("generation") or {}).get("voice_lite") or {}, enabled=False)))
    path = a.examples or vs.get("examples_file")
    if not path:
        print("no examples file: pass --examples or set generation.voice_lite.examples_file", file=sys.stderr)
        return 2
    n = a.n or int(vs["copy_guard_ngram"])
    if not 4 <= n <= 12:
        raise ValueError("--n must be between 4 and 12")
    text = sys.stdin.read() if a.file == "-" else pathlib.Path(a.file).read_text()
    hits = vl.find_copies(text, vl.load_examples(path), n)
    if not hits:
        print(f"copy-check: no run of {n}+ words (or whole example) copied from your style examples")
        return 0
    for h in hits:   # report position and length only: never echo the example text back
        print(f"FLAG copied {h['words']} words from example #{h['example'] + 1} at chars {h['start']}-{h['end']}")
    return 1


def cmd_calibrate(a, cfg):
    """Pick a not-in-notes TRADE-OFF from the user's own labelled questions (stays local; prints counts only).
    On real notes the scores of answerable and out-of-notes questions overlap, so no threshold separates them:
    calibration only chooses how many out-of-notes questions to catch for how many false blocks.
    File: one JSON object per line: {"query": "...", "answerable": true|false}."""
    from eval.run_eval import tune_threshold

    rows = []
    for n, line in enumerate(pathlib.Path(a.questions).read_text().splitlines(), start=1):
        if line.strip():
            r = json.loads(line)
            if not isinstance(r.get("query"), str) or not isinstance(r.get("answerable"), bool):
                raise ValueError(f"line {n}: need {{'query': str, 'answerable': bool}}")
            rows.append({"id": str(n), "query": r["query"], "answerable": r["answerable"]})
    n_ans = sum(r["answerable"] for r in rows)
    if n_ans < 1 or n_ans == len(rows):
        raise ValueError("need both answerable and not-answerable questions (aim for 30+ and 10+)")
    ret, _ = build_retriever(cfg, a.docs, a.cache, mode="dense")
    runs = {r["id"]: {"best": ret.search(r["query"], mode="dense").best_dense} for r in rows}
    thr, bal = tune_threshold(runs, rows, {r["id"] for r in rows})
    cur = cfg["retrieval"].get("not_in_notes_threshold")

    n_nin = len(rows) - n_ans
    ans = sorted(runs[r["id"]]["best"] for r in rows if r["answerable"])
    nin = sorted(runs[r["id"]]["best"] for r in rows if not r["answerable"])

    def stats(t):
        caught = sum(b < t for b in nin)
        blocked = sum(b < t for b in ans)
        return {"threshold": t, "caught": caught, "of_not_in_notes": n_nin, "false_blocks": blocked,
                "of_answerable": n_ans}

    def spread(v):
        return {"min": round(v[0], 3), "median": round(v[len(v) // 2], 3), "max": round(v[-1], 3)}

    overlap = nin[-1] >= ans[0]
    sug = stats(thr)
    print(json.dumps({
        "questions": len(rows), "best_dense_answerable": spread(ans), "best_dense_not_in_notes": spread(nin),
        "scores_overlap": overlap, "current": stats(cur) if cur is not None else None,
        "suggested": dict(sug, balanced_accuracy=bal),
        "trade_off": f"at {thr}: catches {sug['caught']} of {n_nin} out-of-notes questions and falsely blocks "
                     f"{sug['false_blocks']} of {n_ans} answerable ones",
        "advice": ("scores overlap: no threshold separates them. Consider retrieval.not_in_notes_gate: warn "
                   "(results shown with a warning) or off." if overlap else
                   "scores separate on this set; set retrieval.not_in_notes_threshold in your config"),
    }, indent=1))
    if len(rows) - n_ans < 10 or n_ans < 30:
        print("note: few questions; the suggestion is noisy (aim for 30+ answerable and 10+ not-in-notes)", file=sys.stderr)
    return 0


def _generation_health(cfg, cache):
    """Backend + local model + voice lite status. No network: Ollama/llama-server reachability isn't probed."""
    from plm_kit import answer, local_llm
    from plm_kit.models import local_generator

    g = cfg.get("generation") or {}
    s = local_llm.settings(g)
    spec = local_generator(s["model"])
    vl = g.get("voice_lite") or {}
    gguf = local_llm.backend_settings(g, "llamacpp_python").get("gguf_path") or cache / "gguf" / spec["file"]
    return {"backend": g.get("backend", answer.DEFAULT_BACKEND), "available_backends": answer.available(g),
            "local_model": s["model"], "license": spec["license"],
            "ollama_tag": local_llm.backend_settings(g, "ollama").get("model") or spec["ollama"],
            "gguf_ready": pathlib.Path(gguf).expanduser().is_file(),
            "voice_lite": {"enabled": bool(vl.get("enabled")), "examples_file_set": bool(vl.get("examples_file"))}}


def cmd_health(a, cfg):
    from plm_kit.models import RERANKERS, resolve_embedder

    cache = paths.cache_dir(a.cache)
    name, spec = resolve_embedder(cfg["retrieval"]["dense"])
    rname = cfg["retrieval"]["reranker"]["model"]
    idx = paths.index_dir(cfg["data"].get("index_dir"), a.cache)
    mode, note = resolve_mode(cfg["retrieval"])
    od = (cfg.get("sources") or {}).get("onedrive") or {}
    info = {
        "cache_dir": str(cache),
        "retrieval_mode": {"configured": cfg["retrieval"].get("mode", "fused"), "effective": mode, "note": note},
        "embedder": {"name": name, "hf_id": spec["hf_id"], "revision": spec["revision"],
                     "ready": (paths.onnx_dir(name, cache) / "model.onnx").is_file()},
        "reranker": {"name": rname, "revision": RERANKERS[rname]["revision"],
                     "used": mode == "fused+rerank",
                     "ready": (paths.onnx_dir(rname, cache) / "model.onnx").is_file()},
        "generator_ready": (cache / "qwen3-8b-src" / "config.json").is_file(),
        "generation": _generation_health(cfg, cache),
        "index": {"dir": str(idx), "exists": (idx / "manifest.json").is_file()},
        "fusion_k": cfg["retrieval"]["fusion"]["k"],
        "not_in_notes_threshold": cfg["retrieval"].get("not_in_notes_threshold"),
        "not_in_notes_gate": cfg["retrieval"].get("not_in_notes_gate", "block"),
        "newest_copy": (cfg["retrieval"].get("dedupe") or {}).get("newest_copy", "off"),
        "onedrive": {"enabled": bool(od.get("enabled")), "folders": len(od.get("folders") or []),
                     "client_id_set": bool(os.environ.get("PLM_KIT_ONEDRIVE_CLIENT_ID") or od.get("client_id"))},
    }
    print(json.dumps(info, indent=1))
    return 0


def cmd_draft_check(a, cfg):
    from plm_kit import safety

    text = sys.stdin.read() if a.file == "-" else pathlib.Path(a.file).read_text()
    words = safety.load_words(a.words or paths.REPO / "config" / "safety_words.txt")
    found = safety.check(text, words, allow_urls=not a.flag_urls)
    if not found:
        print("draft-check: nothing flagged (a human still reviews before sending)")
        return 0
    for f in found:
        print(f"FLAG {f['kind']}: {f['match']}")
    return 1


def cmd_onedrive(a, cfg):
    from plm_kit import onedrive

    s = onedrive.settings(cfg)
    if a.action == "status":
        tc = onedrive.token_cache_path(s.get("token_cache"))
        print(json.dumps({"tenant_kind": s["tenant"] if s["tenant"].lower() in onedrive.TENANT_ALIASES else "tenant",
                          "scopes": list(onedrive.SCOPES), "folders": s["folders"], "auth": s["auth"],
                          "token_cache": str(tc), "signed_in_cache": tc.is_file(),
                          "sync_dir": str(onedrive.sync_dir(s, a.cache))}, indent=1))
        return 0
    sess = onedrive.Session(s)
    if a.action == "logout":
        sess.logout()
        print(f"signed out; deleted {sess.cache_path} (synced files stay in {onedrive.sync_dir(s, a.cache)} "
              "until you delete them)")
        return 0
    if a.action == "login":
        sess.token(interactive_ok=True)
        print("signed in (scope: Files.Read). Token cache: " + str(sess.cache_path))
        return 0
    folders = [onedrive.normalize_folder(f) for f in a.folder] if a.folder else s["folders"]
    if not folders:
        print("no folders selected: set sources.onedrive.folders or pass --folder", file=sys.stderr)
        return 2
    if not (cfg.get("sources", {}).get("onedrive") or {}).get("enabled"):
        print("sources.onedrive.enabled is false: the files sync, but `index`/`ask` ignore them until you enable it",
              file=sys.stderr)
    reader = onedrive.GraphReader(lambda: sess.token(interactive_ok=False))
    stats = onedrive.sync(reader, dict(s, folders=folders), onedrive.sync_dir(s, a.cache))
    _, info = build_retriever(cfg, cache=a.cache)
    print(json.dumps({"sync": stats, "index": info}, indent=1))
    return 0


def parser():
    p = argparse.ArgumentParser(prog="plm-kit", description=__doc__.splitlines()[0])
    p.add_argument("--config")
    p.add_argument("--cache", help="user cache dir (default: $PLM_KIT_CACHE or ~/.cache/plm-kit)")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="explicitly fetch pinned models from Hugging Face into the user cache")
    d.add_argument("--components", default="embed,reranker",
                   help="comma list: embed, reranker, gguf (the no-GPU answer model in generation.local.model)")
    d.add_argument("--include-generator", action="store_true", help="also fetch Qwen3-8B (~16 GB)")
    d.add_argument("--yes", action="store_true", help="skip the confirmation prompt (the notice still prints)")
    i = sub.add_parser("index", help="build or refresh the index of your notes (stored in the user cache)")
    i.add_argument("--docs")
    q = sub.add_parser("ask", help="search your notes (excerpts by default)")
    q.add_argument("question")
    q.add_argument("--docs")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--mode", choices=list(MODES), help="override retrieval.mode for this query")
    q.add_argument("--preset", choices=sorted(PRESETS), help="quality = fused+rerank, top_n 20, weight 1.0 (~2 s/query CPU)")
    q.add_argument("--answer", action="store_true", help="generate an answer. Default backend auto: no GPU needed "
                   "(Ollama if running, else llama-cpp-python); transformers_peft is the opt-in CUDA GPU path")
    q.add_argument("--backend", help="override generation.backend for this question: auto, ollama, "
                   "llamacpp_server, llamacpp_python, transformers_peft, none, or any registered backend")
    vlg = q.add_mutually_exclusive_group()
    vlg.add_argument("--voice-lite", dest="voice_lite", action="store_true", default=None,
                     help="answer in your style from generation.voice_lite.examples_file (no training)")
    vlg.add_argument("--no-voice-lite", dest="voice_lite", action="store_false")
    e = sub.add_parser("eval", help="run the synthetic golden-set eval")
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--json")
    e.add_argument("--no-grid", action="store_true", help="skip the reranker grid")
    e.add_argument("--newest-copy", choices=["off", "demote", "drop"], help="override retrieval.dedupe.newest_copy (experimental; drop not recommended)")
    e.add_argument("--dedupe-threshold", type=float, help="override retrieval.dedupe.threshold (shingle Jaccard)")
    e.add_argument("--dedupe-multiplier", type=float, help="override retrieval.dedupe.multiplier (demote mode)")
    e.add_argument("--dedupe-tie-tolerance", type=float, help="override retrieval.dedupe.tie_tolerance_s (seconds)")
    e.add_argument("--fixed-threshold", action="store_true",
                   help="gate at the configured not_in_notes_threshold instead of tuning it on a 70/30 split")
    e.add_argument("--docs", help="notes folder for the eval (default: sample_data/docs)")
    e.add_argument("--gold", help="golden-set JSONL (default: eval/golden_synthetic.jsonl)")
    e.add_argument("--preset", choices=sorted(PRESETS), help="apply a preset's reranker settings to the fused+rerank arm")
    e.add_argument("--modes", help="comma list of arms: bm25,dense,fused,fused+rerank (default: all available)")
    sub.add_parser("health", help="show config, model, and index status")
    st = sub.add_parser("selftest", help="starter test kit: index the bundled sample notes, run the golden questions, "
                        "print a pass/fail scorecard (~1 min on CPU; no network)")
    st.add_argument("--answer", action="store_true", help="also generate a few answers with your no-GPU backend")
    st.add_argument("--backend", help="override generation.backend for --answer (e.g. ollama, llamacpp_server)")
    st.add_argument("--json", help="also write the scorecard as JSON here")
    cc = sub.add_parser("copy-check", help="flag text copied word for word from your voice lite examples")
    cc.add_argument("file")
    cc.add_argument("--examples", help="examples file (default: generation.voice_lite.examples_file)")
    cc.add_argument("--n", type=int, help="words in a row that count as a copy (4-12; default copy_guard_ngram)")
    cal = sub.add_parser("calibrate", help="pick a not-in-notes trade-off from your own labelled questions: prints "
                         "catches and false blocks per threshold (it can't fix overlapping scores)")
    cal.add_argument("questions", help='JSONL: {"query": "...", "answerable": true|false} per line')
    cal.add_argument("--docs")
    c = sub.add_parser("draft-check", help="flag personal data / listed words in an outbound draft")
    c.add_argument("file")
    c.add_argument("--words")
    c.add_argument("--flag-urls", action="store_true")
    o = sub.add_parser("onedrive", help="optional read-only OneDrive source (Microsoft Graph, Files.Read)")
    o.add_argument("action", choices=["login", "sync", "status", "logout"])
    o.add_argument("--folder", action="append", help="OneDrive folder to sync (repeatable; overrides config)")
    m = sub.add_parser("mcp", help="read-only notes server for agents (MCP over stdio; see integrations/)")
    m.add_argument("--docs")
    return p


def cmd_mcp(a, cfg):
    from plm_kit import notes_server

    return notes_server.serve(cfg, a.docs, a.cache)


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        cfg = load_config(a.config)
        return {"download": cmd_download, "index": cmd_index, "ask": cmd_ask, "eval": cmd_eval,
                "health": cmd_health, "calibrate": cmd_calibrate, "draft-check": cmd_draft_check,
                "onedrive": cmd_onedrive, "selftest": cmd_selftest, "copy-check": cmd_copy_check,
                "mcp": cmd_mcp}[a.cmd](a, cfg)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
