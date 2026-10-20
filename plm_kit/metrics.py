"""Retrieval metrics for the synthetic golden set."""


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def hit_at_k(ranked, relevant, k):
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def mrr(ranked, relevant):
    for r, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / r
    return 0.0
