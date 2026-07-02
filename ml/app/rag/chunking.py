from __future__ import annotations

from typing import Protocol

from app.rag.models import Chunk, Monograph


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Monograph) -> list[Chunk]: ...


def chunk_metadata(doc: Monograph) -> dict:
    return {
        "drugs_mentioned": doc.all_drug_names(),
        "drug_class": [doc.drug_class_a, doc.drug_class_b],
        "severity": doc.severity,
    }


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Monograph) -> list[Chunk]:
        md = chunk_metadata(doc)
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        idx = 0
        for section, text in doc.sections.items():
            words = text.split()
            if not words:
                continue
            start = 0
            while start < len(words):
                window = words[start : start + self.chunk_size]
                chunks.append(
                    Chunk(
                        text=" ".join(window),
                        source_doc_id=doc.id,
                        section=section,
                        chunk_index=idx,
                        metadata=dict(md),
                    )
                )
                idx += 1
                if start + self.chunk_size >= len(words):
                    break
                start += step
        return chunks
