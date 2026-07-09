from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.deps import get_llm, get_retriever
from app.main import app
from app.rag.models import Chunk

pytestmark = pytest.mark.api


class FakeRetriever:
    def retrieve(self, query, k):
        return [
            Chunk(text="Increased bleeding risk.", source_doc_id="int_warfarin_ibuprofen",
                  section="summary", chunk_index=0,
                  metadata={"source_url": "http://s", "severity": "high"}),
            Chunk(text="Avoid the combination; prefer acetaminophen.",
                  source_doc_id="int_warfarin_ibuprofen", section="management", chunk_index=1,
                  metadata={"severity": "high"}),
        ]


class FakeLLM:
    def generate(self, prompt):
        return "Combining them raises bleeding risk."
    def stream(self, prompt):
        yield "Combining them raises bleeding risk."


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_interactions_check_enhanced_rag_backed(client):
    r = client.post("/interactions/check-enhanced",
                    json={"drugA": "warfarin", "drugB": "ibuprofen", "patientContext": {"age": 70}})
    assert r.status_code == 200
    body = r.json()
    assert body["severity"] == "severe"   # corpus "high" -> dashboard "severe"
    assert body["description"] == "Combining them raises bleeding risk."
    assert body["recommendation"].startswith("Avoid the combination")
    assert body["method"] == "rag"
    assert body["sources"] == ["http://s"]


def test_interactions_check_non_enhanced_rag_backed(client):
    # legacy /interactions/check (no patientContext) must still be served (Express calls it)
    r = client.post("/interactions/check", json={"drugA": "warfarin", "drugB": "ibuprofen"})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "rag"
    assert body["severity"] == "severe"
    assert body["description"] == "Combining them raises bleeding risk."


def test_drugs_served_from_corpus():
    # no override: reads the real committed corpus (offline)
    with TestClient(app) as c:
        r = c.get("/drugs")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0 and len(body["drugs"]) == body["count"]
    names = {d["name"] for d in body["drugs"]}
    assert "warfarin" in names


def test_drug_detail_404_for_unknown():
    with TestClient(app) as c:
        r = c.get("/drugs/notadrug")
    assert r.status_code == 404
