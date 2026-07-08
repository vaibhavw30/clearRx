from __future__ import annotations

import numpy as np

from app.rag.hybrid import HybridRerankRetriever, convex_scale
from app.rag.vectorstore import Match


class FakeEmbedder:
    dimension = 2
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])
    def embed_query(self, text):
        return np.array([1.0, 1.0])


class FakeSparse:
    def encode_query(self, text):
        return {"indices": [3], "values": [2.0]}


class FakeStore:
    def __init__(self):
        self.q = None
    def query(self, dense, top_k, flt, namespace, sparse=None):
        self.q = dict(dense=dense, top_k=top_k, namespace=namespace, sparse=sparse)
        return [Match(id=f"d{i}::s::0", score=1.0,
                      metadata={"chunk_text": f"t{i}", "source_doc_id": f"d{i}",
                                "section": "s", "chunk_index": 0}) for i in range(3)]


class ReverseReranker:
    def rerank(self, query, docs, top_n):
        return list(range(len(docs)))[::-1][:top_n]


def test_convex_scale_weights_both_sides():
    d, s = convex_scale([1.0, 1.0], {"indices": [3], "values": [2.0]}, 0.75)
    assert d == [0.75, 0.75]
    assert s == {"indices": [3], "values": [0.5]}  # 2.0 * (1 - 0.75)


def test_retrieve_queries_top_k_scales_sparse_and_reranks():
    store = FakeStore()
    r = HybridRerankRetriever(FakeEmbedder(), FakeSparse(), store, "hybrid",
                              reranker=ReverseReranker(), alpha=0.5, top_k=50)
    chunks = r.retrieve("q", 2)
    assert store.q["top_k"] == 50 and store.q["namespace"] == "hybrid"
    assert store.q["sparse"] == {"indices": [3], "values": [1.0]}  # 2.0 * 0.5
    assert [c.source_doc_id for c in chunks] == ["d2", "d1"]  # reversed, top 2


def test_retrieve_without_reranker_truncates_in_store_order():
    store = FakeStore()
    r = HybridRerankRetriever(FakeEmbedder(), FakeSparse(), store, "hybrid",
                              reranker=None, alpha=0.5, top_k=50)
    assert [c.source_doc_id for c in r.retrieve("q", 2)] == ["d0", "d1"]
