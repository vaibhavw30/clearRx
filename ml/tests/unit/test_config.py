from __future__ import annotations

from app.config import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.embedding_dim == 1024
    assert s.pinecone_namespace == "curated"
    assert s.corpus_dir.endswith("corpus")


def test_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    monkeypatch.setenv("JUDGE_MODEL", "sonnet")
    s = Settings()
    assert s.embedding_dim == 384
    assert s.judge_model == "sonnet"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


def test_phase1b_settings_defaults(monkeypatch):
    from app.config import Settings
    for var in ["PINECONE_API_KEY", "OPENAI_API_KEY", "OLLAMA_BASE_URL"]:
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.embedding_dim == 1024
    assert s.pinecone_metric == "dotproduct"
    assert s.pinecone_cloud == "aws"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.pinecone_api_key == ""      # safe default, no env needed for unit tests
    assert s.judge_provider == "ollama"


def test_pinecone_key_read_from_env(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("PINECONE_API_KEY", "pc-xyz")
    assert Settings().pinecone_api_key == "pc-xyz"


def test_phase2_chunking_settings_defaults():
    from app.config import Settings
    s = Settings()
    assert s.chunk_recursive_size == 300
    assert s.chunk_recursive_overlap == 60
    assert s.semantic_threshold_percentile == 85.0


def test_phase3_hybrid_rerank_settings_defaults():
    from app.config import Settings
    s = Settings()
    assert s.rerank_provider == "local"
    assert s.rerank_model_local == "BAAI/bge-reranker-base"
    assert s.cohere_rerank_model == "rerank-english-v3.0"
    assert s.hybrid_alphas == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert s.hybrid_top_k == 50
    assert s.hybrid_namespace == "hybrid"
    assert s.bm25_params_path.endswith("bm25_params.json")
