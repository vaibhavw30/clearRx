from __future__ import annotations

from app.answer import build_interaction_response, citations_from_chunks, sse_frame
from app.rag.models import Chunk


def _chunk(doc, section, url=None, severity="high", text="t"):
    md = {"source_url": url, "severity": severity} if url else {"severity": severity}
    return Chunk(text=text, source_doc_id=doc, section=section, chunk_index=0, metadata=md)


def test_citations_distinct_first_seen_with_url():
    chunks = [
        _chunk("int_a", "summary", url="http://x"),
        _chunk("int_a", "management"),   # same doc -> collapsed
        _chunk("int_b", "summary", url="http://y"),
    ]
    cites = citations_from_chunks(chunks)
    assert [c.source_doc_id for c in cites] == ["int_a", "int_b"]
    assert cites[0].url == "http://x" and cites[0].section == "summary"


def test_sse_frame_with_and_without_event():
    assert sse_frame("hello") == "data: hello\n\n"
    assert sse_frame("[1]", event="citations") == "event: citations\ndata: [1]\n\n"


def test_build_interaction_response_maps_from_chunks():
    chunks = [
        _chunk("int_warfarin_ibuprofen", "summary", url="http://s", severity="high"),
        _chunk("int_warfarin_ibuprofen", "management", text="Avoid the combination."),
    ]
    r = build_interaction_response(chunks, answer="Increased bleeding risk.")
    assert r.severity == "high"
    assert r.description == "Increased bleeding risk."
    assert r.recommendation == "Avoid the combination."   # from the management-section chunk
    assert r.sources == ["http://s"]                       # url preferred over id
    assert r.method == "rag" and r.confidence > 0.0


def test_build_interaction_response_no_chunks_is_safe():
    r = build_interaction_response([], answer="")
    assert r.severity == "unknown" and r.sources == [] and r.confidence == 0.0
    assert "Consult" in r.recommendation and r.method == "rag"
