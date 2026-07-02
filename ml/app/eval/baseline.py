from __future__ import annotations

import re
from typing import Optional

from app.rag.chunking import Chunker, FixedSizeChunker
from app.rag.models import Chunk, Monograph


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


class KeywordBaseline:
    """Honest stand-in for today's system, which has no query retrieval at
    all: naive keyword overlap over corpus chunks + a deterministic,
    LLM-free templated answer."""

    def __init__(self, docs: list[Monograph], chunker: Optional[Chunker] = None) -> None:
        self.chunker = chunker or FixedSizeChunker(chunk_size=512, overlap=0)
        self.chunks: list[Chunk] = []
        for doc in docs:
            self.chunks.extend(self.chunker.chunk(doc))

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        q = _tokens(query)
        scored = [(len(q & _tokens(c.text)), c.chunk_index, c) for c in self.chunks]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for score, _, c in scored[:k] if score > 0]

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant interaction information found."
        body = " ".join(c.text for c in chunks)
        return f"Based on the available information: {body}"
