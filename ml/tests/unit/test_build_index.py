from __future__ import annotations

import numpy as np

from app.ingest.build_index import build_records
from app.rag.chunking import FixedSizeChunker
from app.rag.models import Monograph


class FakeEmbedder:
    dimension = 3

    def embed(self, texts):
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)

    def embed_query(self, text):
        return np.zeros(3, dtype=np.float32)


def _doc():
    return Monograph(
        id="int_a_b", drug_a="a", drug_b="b", drug_class_a="x", drug_class_b="y",
        severity="high",
        sections={"summary": "a and b interact", "management": "avoid the combination"},
    )


def test_build_records_ids_and_metadata():
    recs = build_records([_doc()], FixedSizeChunker(chunk_size=512, overlap=0), FakeEmbedder())
    assert len(recs) == 2  # one chunk per non-empty section
    ids = {r.id for r in recs}
    assert "int_a_b::summary::0" in ids
    summary = next(r for r in recs if r.id == "int_a_b::summary::0")
    assert summary.metadata["chunk_text"] == "a and b interact"
    assert summary.metadata["source_doc_id"] == "int_a_b"
    assert summary.metadata["section"] == "summary"
    assert summary.metadata["severity"] == "high"
    assert "a" in summary.metadata["drugs_mentioned"]
    assert len(summary.values) == 3  # embedding attached
