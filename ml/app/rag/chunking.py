from __future__ import annotations

import re

import numpy as np
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


class RecursiveChunker:
    """Recursive character splitting over the whole monograph using
    LangChain's RecursiveCharacterTextSplitter. Sections are concatenated
    with their names as light headers so the splitter can choose boundaries
    across the document rather than being confined to one section."""

    name = "recursive"

    def __init__(self, chunk_size: int = 300, overlap: int = 60, *, splitter=None) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = splitter

    def _get_splitter(self):
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # lazy

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return self._splitter

    def chunk(self, doc: Monograph) -> list[Chunk]:
        md = chunk_metadata(doc)
        text = "\n\n".join(
            f"{section}. {body}" for section, body in doc.sections.items() if body.strip()
        )
        pieces = [p for p in self._get_splitter().split_text(text) if p.strip()]
        return [
            Chunk(
                text=piece,
                source_doc_id=doc.id,
                section="document",
                chunk_index=i,
                metadata=dict(md),
            )
            for i, piece in enumerate(pieces)
        ]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Semantic-breakpoint chunking over the whole monograph. Sentences are
    embedded; a boundary is placed wherever the cosine distance between
    consecutive sentences reaches the configured percentile of all such
    distances. Embeddings are assumed L2-normalized, so cosine distance is
    1 - dot product."""

    name = "semantic"

    def __init__(self, embedder, threshold_percentile: float = 85.0) -> None:
        self.embedder = embedder
        self.threshold_percentile = threshold_percentile

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    def chunk(self, doc: Monograph) -> list[Chunk]:
        md = chunk_metadata(doc)
        text = " ".join(body for body in doc.sections.values() if body.strip())
        sentences = self._sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [Chunk(text=sentences[0], source_doc_id=doc.id, section="document",
                          chunk_index=0, metadata=dict(md))]

        vecs = self.embedder.embed(sentences)
        distances = [
            1.0 - float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(sentences) - 1)
        ]
        threshold = float(np.percentile(distances, self.threshold_percentile))

        groups: list[list[str]] = [[sentences[0]]]
        for i, dist in enumerate(distances):
            if threshold > 0 and dist >= threshold:
                groups.append([sentences[i + 1]])
            else:
                groups[-1].append(sentences[i + 1])

        texts = [" ".join(g).strip() for g in groups]
        return [
            Chunk(text=t, source_doc_id=doc.id, section="document", chunk_index=i,
                  metadata=dict(md))
            for i, t in enumerate(texts)
            if t
        ]
