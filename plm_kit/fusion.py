"""Reciprocal Rank Fusion over ranked lists of doc ids."""


def rrf(rankings, k=60, weights=None):
    """rankings: list of ranked id lists. Returns ids sorted by sum(w / (k + rank))."""
    weights = weights or [1.0] * len(rankings)
    score = {}
    for w, ranked in zip(weights, rankings):
        for r, doc_id in enumerate(ranked, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + w / (k + r)
    return sorted(score, key=lambda d: (-score[d], d))
