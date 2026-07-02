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
