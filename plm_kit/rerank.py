"""Optional reranker plug-in. Off by default; when off, nothing is imported, downloaded, or loaded.
Any failure leaves the fused order unchanged."""
import os

from plm_kit.models import RERANKERS
from plm_kit.truncate import DEFAULT_MAX_TOKENS, Truncator, cap_backend_tokenizer

_STATE = {"fn": None, "tried": False, "error": None}


def enabled(cfg):
    env = os.environ.get("PLM_KIT_RERANKER", "").lower()
    if env in ("off", "0", "false"):
        return False
    on = cfg.get("enabled", False) or env in ("on", "1", "true")
    return bool(on and cfg.get("top_n", 0) and cfg.get("weight", 0))


def make_scorer(enc, truncator, batch_size=16):
    """(query, docs) -> scores, with every (query, doc) pair cut to the token budget before scoring."""
    def score(q, docs):
        pairs = [truncator.cut_pair(q, d) for d in docs]
        qq = pairs[0][0] if pairs else q
        return [float(x) for x in enc.rerank(qq, [d for _, d in pairs], batch_size=batch_size)]
    return score


def _load(spec, onnx_dir, threads=None):
    """Load a locally exported ONNX cross-encoder via fastembed 0.9.0 (no network: specific_model_path)."""
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    # fastembed 0.9.0: add_custom_model(model, sources, model_file="onnx/model.onnx", description="",
    #                                   license="", size_in_gb=0.0, additional_files=None)
    try:
        TextCrossEncoder.add_custom_model(model=spec["hf_id"], sources=ModelSource(hf=spec["hf_id"]),
                                          model_file="model.onnx", license=spec["license"],
                                          size_in_gb=spec["size_in_gb"])
    except ValueError:  # already registered in this process
        pass
    return TextCrossEncoder(model_name=spec["hf_id"], specific_model_path=onnx_dir, threads=threads)


def get_reranker(cfg):
    """Return a (query, docs) -> list[float] scorer, or None (stage skipped).
    Disabled -> returns before importing fastembed or touching the filesystem/network."""
    if not enabled(cfg):
        return None
    if _STATE["tried"]:
        return _STATE["fn"]
    _STATE["tried"] = True
    try:
        spec = RERANKERS[cfg["model"]]
        onnx_dir = cfg.get("onnx_dir")
        if not onnx_dir or not os.path.isfile(os.path.join(onnx_dir, "model.onnx")):
            raise FileNotFoundError("reranker ONNX not exported; run scripts/export_onnx.py")
        enc = _load(spec, onnx_dir, threads=cfg.get("threads"))
        max_tokens = cfg.get("max_tokens") or DEFAULT_MAX_TOKENS
        cap_backend_tokenizer(enc, max_tokens)
        _STATE["fn"] = make_scorer(enc, Truncator(max_tokens, tokenizer_dir=onnx_dir))
    except Exception as e:  # noqa: BLE001 - never break search; log type only (no paths or content)
        _STATE["error"] = type(e).__name__
    return _STATE["fn"]


def blend(query, order, texts, cfg, scorer, archived=frozenset(), archive_weight=1.0, multipliers=None):
    """Rerank the top_n of `order` (fused ranking of ids):
    score = 1/(k + fused_rank) + weight/(k + ce_rank), times the chunk's stale-copy multiplier
    (`multipliers` {id: m}; or the older archived/archive_weight pair).
    Returns the new order; on any scorer error returns `order` unchanged."""
    n, w, k = cfg.get("top_n", 0), cfg.get("weight", 0.0), cfg.get("k", 20)
    if not scorer or not n or not w or not order:
        return list(order)
    head, tail = list(order[:n]), list(order[n:])
    try:
        scores = scorer(query, [texts[i] for i in head])
        if len(scores) != len(head):
            return list(order)
    except Exception:  # noqa: BLE001 - fail closed
        return list(order)
    ce_rank = {i: r for r, (i, _) in enumerate(sorted(zip(head, scores), key=lambda x: -x[1]), start=1)}
    def mult(i):
        return multipliers.get(i, 1.0) if multipliers is not None else (archive_weight if i in archived else 1.0)

    final = {i: (1.0 / (k + r) + w / (k + ce_rank[i])) * mult(i)
             for r, i in enumerate(head, start=1)}
    return sorted(head, key=lambda i: -final[i]) + tail
