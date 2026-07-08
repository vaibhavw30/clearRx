from __future__ import annotations

from app.rag.models import Chunk
from app.rag.retrieval import chunks_from_matches


def convex_scale(dense: list[float], sparse: dict, alpha: float) -> tuple:
    """Weight dense vs sparse for a Pinecone hybrid query: dense * alpha,
    sparse values * (1 - alpha). Embeddings are assumed dot-product scored."""
    scaled_dense = [float(v) * alpha for v in dense]
    scaled_sparse = {
        "indices": list(sparse["indices"]),
        "values": [float(v) * (1.0 - alpha) for v in sparse["values"]],
    }
    return scaled_dense, scaled_sparse


class HybridRerankRetriever:
    """Two-stage retrieval: hybrid (dense + BM25 sparse) recall of top_k, then
    optional cross-encoder rerank to the final k. Retrieval-stage only —
    exposes `.retrieve` like the dense pipeline."""

    def __init__(self, embedder, sparse_encoder, store, namespace, *,
                 reranker=None, alpha: float = 0.5, top_k: int = 50) -> None:
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.store = store
        self.namespace = namespace
        self.reranker = reranker
        self.alpha = alpha
        self.top_k = top_k

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        dense = self.embedder.embed_query(query)
        sparse = self.sparse_encoder.encode_query(query)
        scaled_dense, scaled_sparse = convex_scale(list(dense), sparse, self.alpha)
        matches = self.store.query(
            scaled_dense, top_k=self.top_k, flt=None,
            namespace=self.namespace, sparse=scaled_sparse,
        )
        cands = chunks_from_matches(matches)
        if self.reranker is None:
            return cands[:k]
        order = self.reranker.rerank(query, [c.text for c in cands], top_n=k)
        return [cands[i] for i in order]
