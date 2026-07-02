from __future__ import annotations

from typing import Protocol

from app.rag.models import Chunk


class RagPipeline(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...
    def generate(self, query: str, chunks: list[Chunk]) -> str: ...
