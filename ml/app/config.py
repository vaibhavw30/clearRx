from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

_DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    corpus_dir: str = os.path.normpath(os.path.join(_DATA, "corpus"))
    eval_dir: str = os.path.normpath(os.path.join(_DATA, "eval"))
    reports_dir: str = os.path.normpath(os.path.join(_DATA, os.pardir, "eval", "reports"))

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    pinecone_index: str = "clearrx-drug-interactions"
    pinecone_namespace: str = "curated"

    gen_model: str = "llama3.1"
    judge_provider: str = "ollama"
    judge_model: str = "llama3.1"

    pinecone_api_key: str = ""
    pinecone_metric: str = "dotproduct"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""

    chunk_recursive_size: int = 300
    chunk_recursive_overlap: int = 60
    semantic_threshold_percentile: float = 85.0
    rerank_provider: str = "local"
    rerank_model_local: str = "BAAI/bge-reranker-base"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"
    hybrid_alphas: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    hybrid_top_k: int = 50
    hybrid_namespace: str = "hybrid"
    bm25_params_path: str = os.path.normpath(os.path.join(_DATA, "bm25_params.json"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
