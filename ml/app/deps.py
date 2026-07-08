from __future__ import annotations

from app.config import get_settings
from app.rag.embeddings import BGEEmbedder
from app.rag.generator import OllamaClient
from app.rag.pipeline import DenseRagPipeline
from app.rag.vectorstore import PineconeStore


def get_retriever():
    """Dense retriever for query-time retrieval. `.retrieve` is all the routes
    use; `llm=None` is safe because DenseRagPipeline.retrieve ignores it.
    Overridden with a fake in tests."""
    s = get_settings()
    embedder = BGEEmbedder(s.embedding_model, s.embedding_dim)
    store = PineconeStore(s.pinecone_api_key, s.pinecone_index)
    return DenseRagPipeline(embedder, store, llm=None, namespace=s.pinecone_namespace)


def get_llm():
    """LLM client for generation/streaming. Overridden with a fake in tests."""
    s = get_settings()
    return OllamaClient(s.ollama_base_url, s.gen_model)
