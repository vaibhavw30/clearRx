from __future__ import annotations

from typing import Protocol

import numpy as np

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class BGEEmbedder:
    """BGE sentence-transformer embedder. Vectors are L2-normalized so a
    Pinecone dotproduct index behaves as cosine. Queries get the BGE
    retrieval instruction; documents do not."""

    def __init__(self, model_name: str, dimension: int = 1024, *, model=None) -> None:
        self.dimension = dimension
        if model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            model = SentenceTransformer(model_name)
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vecs = self.model.encode(
            [QUERY_INSTRUCTION + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)[0]
