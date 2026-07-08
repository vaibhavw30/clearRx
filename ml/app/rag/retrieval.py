from __future__ import annotations

from app.rag.models import Chunk


def chunks_from_matches(matches) -> list[Chunk]:
    """Rebuild Chunks from vector-store Matches. Retrieval stores the chunk
    text and provenance in metadata, so every retriever reconstructs Chunks
    the same way."""
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
