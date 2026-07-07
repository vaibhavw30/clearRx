from __future__ import annotations

from typing import Protocol

from app.rag.embeddings import Embedder
from app.rag.generator import LLMClient
from app.rag.models import Chunk
from app.rag.vectorstore import VectorStore


class RagPipeline(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...
    def generate(self, query: str, chunks: list[Chunk]) -> str: ...


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.source_doc_id} / {c.section}] {c.text}")
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = build_context(chunks)
    return (
        "You are a careful clinical drug-interaction assistant. Answer ONLY from the "
        "context below. If the context does not cover the drugs asked about, say there "
        "is no interaction information available rather than guessing. Cite the source "
        "document id in square brackets after each claim. This is not medical advice.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


class DenseRagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        namespace: str,
        k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.namespace = namespace
        self.k = k

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        vec = self.embedder.embed_query(query)
        matches = self.store.query(
            vec.tolist(), top_k=k, flt=None, namespace=self.namespace
        )
        chunks: list[Chunk] = []
        for m in matches:
            md = m.metadata
            chunks.append(
                Chunk(
                    text=md.get("chunk_text", ""),
                    source_doc_id=md.get("source_doc_id", ""),
                    section=md.get("section", ""),
                    chunk_index=int(md.get("chunk_index", 0)),
                    metadata=md,
                )
            )
        return chunks

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant interaction information found in the corpus."
        return self.llm.generate(build_prompt(query, chunks))
