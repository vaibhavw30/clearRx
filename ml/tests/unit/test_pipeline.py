from __future__ import annotations

import numpy as np

from app.rag.models import Chunk
from app.rag.pipeline import DenseRagPipeline, build_prompt
from app.rag.vectorstore import Match


class FakeEmbedder:
    dimension = 2

    def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text):
        return np.array([0.6, 0.8], dtype=np.float32)


class FakeStore:
    def __init__(self):
        self.last = {}

    def upsert(self, records, namespace):
        ...

    def query(self, dense, top_k, flt, namespace):
        self.last = {"dense": list(dense), "top_k": top_k, "namespace": namespace}
        return [
            Match(id="int_warfarin_ibuprofen::summary::0", score=0.9,
                  metadata={"chunk_text": "increased bleeding risk",
                            "source_doc_id": "int_warfarin_ibuprofen",
                            "section": "summary", "chunk_index": 0})
        ]


class FakeLLM:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return "There is an increased bleeding risk [int_warfarin_ibuprofen]."

    def stream(self, prompt):
        yield self.generate(prompt)


def test_retrieve_embeds_query_and_maps_matches_to_chunks():
    store = FakeStore()
    pipe = DenseRagPipeline(FakeEmbedder(), store, FakeLLM(), namespace="curated")
    chunks = pipe.retrieve("advil with coumadin?", k=5)
    assert store.last["top_k"] == 5
    assert store.last["namespace"] == "curated"
    assert np.allclose(store.last["dense"], [0.6, 0.8])    # the query embedding, not a doc embedding
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.text == "increased bleeding risk"
    assert c.source_doc_id == "int_warfarin_ibuprofen"
    assert c.section == "summary"


def test_generate_includes_context_and_query_in_prompt():
    llm = FakeLLM()
    pipe = DenseRagPipeline(FakeEmbedder(), FakeStore(), llm, namespace="curated")
    chunk = Chunk(text="increased bleeding risk", source_doc_id="int_warfarin_ibuprofen",
                  section="summary", chunk_index=0)
    answer = pipe.generate("advil with coumadin?", [chunk])
    assert "increased bleeding risk" in llm.prompt        # context present
    assert "advil with coumadin?" in llm.prompt            # query present
    assert answer.startswith("There is an increased bleeding risk")


def test_build_prompt_cites_source_ids():
    chunk = Chunk(text="risk", source_doc_id="int_warfarin_ibuprofen",
                  section="summary", chunk_index=0)
    prompt = build_prompt("q?", [chunk])
    assert "int_warfarin_ibuprofen" in prompt
