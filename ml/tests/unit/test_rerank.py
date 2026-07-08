from __future__ import annotations

from app.config import Settings
from app.rag.rerank import CohereReranker, LocalReranker, build_reranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.seen = None
    def predict(self, pairs):
        self.seen = pairs
        return self.scores


def test_local_reranker_orders_by_score_and_truncates():
    ce = FakeCrossEncoder([0.1, 0.9, 0.5])
    idx = LocalReranker("m", model=ce).rerank("q", ["a", "b", "c"], top_n=2)
    assert idx == [1, 2]  # 0.9(b) > 0.5(c) > 0.1(a); top 2 -> indices 1, 2
    assert ce.seen == [("q", "a"), ("q", "b"), ("q", "c")]


def test_local_reranker_empty_docs():
    assert LocalReranker("m", model=FakeCrossEncoder([])).rerank("q", [], 5) == []


class _R:
    def __init__(self, i):
        self.index = i

class FakeCohereClient:
    def __init__(self, idxs):
        self.idxs = idxs
        self.called = None
    def rerank(self, query, documents, top_n, model):
        self.called = (query, documents, top_n, model)
        return type("Res", (), {"results": [_R(i) for i in self.idxs[:top_n]]})()


def test_cohere_reranker_maps_result_indices():
    c = FakeCohereClient([2, 0, 1])
    out = CohereReranker("key", "rerank-english-v3.0", client=c).rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [2, 0]
    assert c.called[3] == "rerank-english-v3.0"


def test_build_reranker_selects_provider():
    local, coh = object(), object()
    assert build_reranker(Settings(rerank_provider="local"), local=local, cohere=coh) is local
    assert build_reranker(Settings(rerank_provider="cohere"), local=local, cohere=coh) is coh


def test_build_reranker_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        build_reranker(Settings(rerank_provider="bogus"))
