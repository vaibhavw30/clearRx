from __future__ import annotations

import math


def retrieval_coverage(expected_topics: list[str], retrieved_texts: list[str]) -> float:
    if not expected_topics:
        return 0.0
    blob = " \n ".join(t.lower() for t in retrieved_texts)
    hits = sum(1 for topic in expected_topics if topic.lower() in blob)
    return hits / len(expected_topics)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for r in topk if r in relevant_ids) / len(topk)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(topk & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(retrieved_ids[:k], start=1):
        if r in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(s[int(rank)])
    frac = rank - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)
