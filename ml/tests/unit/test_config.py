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
