from __future__ import annotations

from app.eval.aggregate import distinct_doc_ids, mean
from app.rag.models import Chunk


def test_mean_of_values_and_empty():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0


def test_distinct_doc_ids_preserves_first_seen_order():
    chunks = [
        Chunk(text="a", source_doc_id="d2", section="s", chunk_index=0),
        Chunk(text="b", source_doc_id="d1", section="s", chunk_index=1),
        Chunk(text="c", source_doc_id="d2", section="s", chunk_index=2),
    ]
    assert distinct_doc_ids(chunks) == ["d2", "d1"]
