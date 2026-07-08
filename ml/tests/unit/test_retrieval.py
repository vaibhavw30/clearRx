from __future__ import annotations

from app.rag.retrieval import chunks_from_matches
from app.rag.vectorstore import Match


def test_chunks_from_matches_reads_metadata():
    m = Match(id="d::s::2", score=1.0, metadata={
        "chunk_text": "t", "source_doc_id": "d", "section": "s",
        "chunk_index": 2, "severity": "high"})
    c = chunks_from_matches([m])[0]
    assert c.text == "t" and c.source_doc_id == "d" and c.section == "s"
    assert c.chunk_index == 2 and c.metadata["severity"] == "high"


def test_chunks_from_matches_defaults_missing_fields():
    c = chunks_from_matches([Match(id="x", score=0.0, metadata={})])[0]
    assert c.text == "" and c.source_doc_id == "" and c.chunk_index == 0
