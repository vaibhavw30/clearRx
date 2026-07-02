from __future__ import annotations

import math

from app.eval.metrics import (
    ndcg_at_k, percentile, precision_at_k, recall_at_k,
    reciprocal_rank, retrieval_coverage,
)


def test_retrieval_coverage():
    topics = ["NSAID anticoagulant", "monitor INR"]
    texts = ["... nsaid anticoagulant bleeding ...", "... watch closely ..."]
    assert retrieval_coverage(topics, texts) == 0.5
    assert retrieval_coverage([], texts) == 0.0


def test_precision_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5   # 2 of top 4
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3    # 2 of 3 relevant
    assert precision_at_k(retrieved, relevant, 2) == 0.5   # 1 of top 2


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5   # first hit at rank 2
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_at_k():
    # relevant at ranks 1 and 3 -> DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # ideal (2 relevant) IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309
    got = ndcg_at_k(["a", "x", "c"], {"a", "c"}, 3)
    assert math.isclose(got, 1.5 / (1 + 1 / math.log2(3)), rel_tol=1e-9)
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([42], 95) == 42.0
    assert percentile([], 50) == 0.0
