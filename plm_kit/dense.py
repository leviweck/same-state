"""Dense retrieval: a pinned embedding model (default granite-embedding-english-r2) as local ONNX on CPU via fastembed 0.9.0."""
import numpy as np

from plm_kit.truncate import DEFAULT_MAX_TOKENS, Truncator, cap_backend_tokenizer

BATCH_SIZE = 32


class FastembedEmbedder:
    """Loads a user-exported ONNX model (scripts/export_onnx.py); never downloads at query time."""

    def __init__(self, spec, onnx_dir, threads=None, max_tokens=DEFAULT_MAX_TOKENS, batch_size=BATCH_SIZE):
        # Task prefixes: only if the model card specifies them (spec "query_prefix"/"document_prefix").
        # granite-embedding-english-r2's card encodes queries and passages the same way, so both are "".
        """spec: a dict from plm_kit.models.resolve_embedder (registry entry or config `dense.custom`).
        Inputs are capped at max_tokens; texts are embedded batch_size at a time into one preallocated matrix."""
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        try:
            TextEmbedding.add_custom_model(
                model=spec["hf_id"], pooling=PoolingType.CLS if spec["pooling"] == "cls" else PoolingType.MEAN,
                normalization=spec["normalize"], sources=ModelSource(hf=spec["hf_id"]), dim=spec["dim"],
                model_file="model.onnx", license=spec["license"], size_in_gb=spec["size_in_gb"])
        except ValueError:  # already registered in this process
            pass
        self._m = TextEmbedding(model_name=spec["hf_id"], specific_model_path=onnx_dir, threads=threads)
        self.dim, self.batch_size = int(spec["dim"]), int(batch_size)
        self.query_prefix, self.document_prefix = spec.get("query_prefix", ""), spec.get("document_prefix", "")
        self.truncator = Truncator(max_tokens, tokenizer_dir=onnx_dir)
        cap_backend_tokenizer(self._m, max_tokens)

    def embed(self, texts, kind="document"):
        pre = self.query_prefix if kind == "query" else self.document_prefix
        return embed_batched(lambda b: self._m.embed(b, batch_size=self.batch_size),
                             (pre + t for t in texts) if pre else texts, self.dim, self.truncator, self.batch_size)


def embed_batched(embed_fn, texts, dim, truncator, batch_size=BATCH_SIZE):
    """Truncate each text, embed batch_size at a time, and write rows straight into one float32 matrix
    (no list of per-text arrays, no second copy). Rows are L2-normalized in place."""
    texts = list(texts)
    out = np.empty((len(texts), dim), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = [truncator.cut(t) for t in texts[start:start + batch_size]]
        for i, v in enumerate(embed_fn(batch)):
            out[start + i] = v
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    out /= norms
    return out


def _embed_query(embedder, query):
    try:
        return embedder.embed([query], kind="query")[0]
    except TypeError:  # embedders without a `kind` argument (e.g. test fakes)
        return embedder.embed([query])[0]


class DenseIndex:
    def __init__(self, chunks, embedder, matrix=None):
        self.embedder = embedder
        if matrix is not None:
            self.matrix = matrix
        else:
            self.matrix = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 1), np.float32)

    def search(self, query, limit=50, return_all=False):
        """Return (chunk ids best first, cosine scores aligned) [+ the full cosine vector if return_all]."""
        if not len(self.matrix):
            return ([], [], None) if return_all else ([], [])
        q = _embed_query(self.embedder, query)
        sims = self.matrix @ q
        order = np.argsort(-sims, kind="stable")[:limit]
        res = [int(i) for i in order], [float(sims[i]) for i in order]
        return (*res, sims) if return_all else res
