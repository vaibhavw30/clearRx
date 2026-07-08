from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.deps import get_llm, get_retriever
from app.main import app
from app.rag.models import Chunk

pytestmark = pytest.mark.api


class FakeRetriever:
    def retrieve(self, query, k):
        return [Chunk(text="increased bleeding risk", source_doc_id="int_warfarin_ibuprofen",
                      section="summary", chunk_index=0,
                      metadata={"source_url": "http://x", "severity": "high"})]


class FakeLLM:
    def generate(self, prompt):
        return "There is an increased bleeding risk. [int_warfarin_ibuprofen]"
    def stream(self, prompt):
        yield "There is an increased "
        yield "bleeding risk. [int_warfarin_ibuprofen]"


class EmptyRetriever:
    def retrieve(self, query, k):
        return []


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_ok():
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_query_returns_answer_and_grounded_citations(client):
    r = client.post("/query", json={"query": "warfarin ibuprofen"})
    assert r.status_code == 200
    body = r.json()
    assert "bleeding risk" in body["answer"]
    assert body["citations"][0]["source_doc_id"] == "int_warfarin_ibuprofen"
    assert body["citations"][0]["url"] == "http://x"


def test_query_no_chunks_returns_no_info_answer():
    app.dependency_overrides[get_retriever] = lambda: EmptyRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    try:
        r = TestClient(app).post("/query", json={"query": "aspirin water"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json()["citations"] == []
    assert "no interaction information" in r.json()["answer"].lower()
