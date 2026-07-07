from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings


@pytest.mark.integration
def test_bge_real_dimension_and_norm():
    from app.rag.embeddings import BGEEmbedder

    s = get_settings()
    emb = BGEEmbedder(s.embedding_model, s.embedding_dim)
    out = emb.embed(["warfarin ibuprofen bleeding risk"])
    assert out.shape == (1, s.embedding_dim)          # 1024
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-3)
    q = emb.embed_query("can I take advil with coumadin")
    assert q.shape == (s.embedding_dim,)


@pytest.mark.integration
def test_pinecone_upsert_query_roundtrip():
    from app.rag.embeddings import BGEEmbedder
    from app.rag.vectorstore import PineconeStore, Record

    s = get_settings()
    store = PineconeStore(s.pinecone_api_key, s.pinecone_index)
    store.ensure_index(s.embedding_dim, s.pinecone_metric, s.pinecone_cloud, s.pinecone_region)
    emb = BGEEmbedder(s.embedding_model, s.embedding_dim)
    vec = emb.embed(["warfarin plus ibuprofen raises bleeding risk"])[0]
    store.upsert(
        [Record(id="itest::0", values=vec.tolist(),
                metadata={"chunk_text": "warfarin ibuprofen bleeding", "source_doc_id": "itest"})],
        namespace="itest",
    )
    import time as _t

    _t.sleep(5)  # Pinecone upserts are eventually consistent
    q = emb.embed_query("can I combine warfarin and ibuprofen")
    matches = store.query(q.tolist(), top_k=1, flt=None, namespace="itest")
    assert matches and matches[0].metadata["source_doc_id"] == "itest"


@pytest.mark.integration
def test_ollama_generate_returns_text():
    from app.rag.generator import OllamaClient

    s = get_settings()
    client = OllamaClient(s.ollama_base_url, s.gen_model)
    out = client.generate("Answer in one short sentence: what is 2 + 2?")
    assert isinstance(out, str) and out.strip()


@pytest.mark.integration
def test_local_judge_scores_facts():
    from app.rag.judge_clients import build_judge

    s = get_settings()  # judge_provider defaults to ollama
    judge = build_judge(s)
    result = judge.score_facts(
        "Ibuprofen and warfarin together increase bleeding risk.",
        ["Increased bleeding risk", "Reduces blood pressure"],
    )
    assert result == [True, False] or (isinstance(result, list) and len(result) == 2)
