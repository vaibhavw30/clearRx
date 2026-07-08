from __future__ import annotations

from app.schemas import (
    Citation, EnhancedInteractionRequest, InteractionResponse,
    QueryRequest, QueryResponse,
)


def test_query_request_defaults_top_k():
    assert QueryRequest(query="warfarin ibuprofen").top_k == 5


def test_query_response_nests_citations():
    r = QueryResponse(answer="a", citations=[Citation(source_doc_id="int_x")])
    assert r.citations[0].source_doc_id == "int_x"
    assert r.citations[0].url is None


def test_interaction_response_contract_fields():
    r = InteractionResponse(severity="high", description="d", recommendation="r", method="rag")
    assert r.sources == [] and r.confidence == 0.0 and r.method == "rag"


def test_enhanced_interaction_request_patient_context_optional():
    req = EnhancedInteractionRequest(drugA="warfarin", drugB="ibuprofen")
    assert req.patientContext == {}
