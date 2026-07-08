from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.deps import get_llm, get_retriever
from app.main import app
from app.rag.models import Chunk

pytestmark = pytest.mark.api


class FakeRetriever:
    def retrieve(self, query, k):
        return [Chunk(text="bleeding", source_doc_id="int_warfarin_ibuprofen",
                      section="summary", chunk_index=0,
                      metadata={"source_url": "http://x", "severity": "high"})]


class FakeLLM:
    _CHUNKS = ["There is an increased ", "bleeding risk. [int_warfarin_ibuprofen]"]
    def generate(self, prompt):
        return "".join(self._CHUNKS)
    def stream(self, prompt):
        for c in self._CHUNKS:
            yield c


def _tokens_from_sse(text: str) -> str:
    """Reassemble the answer from data: frames, ignoring the citations event and [DONE]."""
    out = []
    for block in text.split("\n\n"):
        if not block or block.startswith("event:"):
            continue
        line = block[len("data: "):] if block.startswith("data: ") else ""
        if line and line != "[DONE]":
            out.append(line)
    return "".join(out)


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_stream_is_event_stream_and_matches_nonstream_answer(client):
    stream = client.post("/query/stream", json={"query": "warfarin ibuprofen"})
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    body = stream.text
    # assembled tokens equal the non-streamed answer
    nonstream = client.post("/query", json={"query": "warfarin ibuprofen"}).json()["answer"]
    assert _tokens_from_sse(body) == nonstream
    # citations event present and grounded; terminal [DONE]
    assert "event: citations" in body
    cite_block = [b for b in body.split("\n\n") if b.startswith("event: citations")][0]
    cites = json.loads(cite_block.split("data: ", 1)[1])
    assert cites[0]["source_doc_id"] == "int_warfarin_ibuprofen"
    assert body.rstrip().endswith("data: [DONE]")
