"""Network-free stand-ins for the ONNX models (deterministic hashed bag-of-words)."""
import hashlib
import re

import numpy as np


class HashEmbedder:
    def __init__(self, dim=256):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
