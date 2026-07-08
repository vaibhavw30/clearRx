from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]: ...


class LocalReranker:
    """Cross-encoder reranker via sentence-transformers. Inject `model` in
    tests to avoid loading a real CrossEncoder."""

    def __init__(self, model_name: str, *, model=None) -> None:
        self.model_name = model_name
        self._model = model

    def _m(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]:
        if not docs:
            return []
        scores = self._m().predict([(query, d) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return order[:top_n]


class CohereReranker:
    """Cohere Rerank. Inject `client` in tests to avoid the SDK/network."""

    def __init__(self, api_key: str, model: str, *, client=None) -> None:
        self.model = model
        if client is None:
            from cohere import Client  # lazy

            client = Client(api_key=api_key)
        self.client = client

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]:
        if not docs:
            return []
        res = self.client.rerank(query=query, documents=docs, top_n=top_n, model=self.model)
        return [r.index for r in res.results]


def build_reranker(settings, *, local=None, cohere=None):
    provider = settings.rerank_provider.lower()
    if provider == "local":
        return local or LocalReranker(settings.rerank_model_local)
    if provider == "cohere":
        return cohere or CohereReranker(settings.cohere_api_key, settings.cohere_rerank_model)
    raise ValueError(f"unknown rerank_provider: {settings.rerank_provider!r}")
