from __future__ import annotations

from typing import Protocol


class SparseEncoder(Protocol):
    def fit(self, corpus_texts: list[str]) -> None: ...
    def encode_documents(self, texts: list[str]) -> list[dict]: ...
    def encode_query(self, text: str) -> dict: ...
    def dump(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


class BM25SparseEncoder:
    """BM25 sparse encoder backed by pinecone-text. Inject `encoder` in tests
    to avoid importing pinecone-text or fitting a real corpus."""

    def __init__(self, *, encoder=None) -> None:
        self._encoder = encoder

    def _enc(self):
        if self._encoder is None:
            from pinecone_text.sparse import BM25Encoder  # lazy

            self._encoder = BM25Encoder()
        return self._encoder

    def fit(self, corpus_texts: list[str]) -> None:
        self._enc().fit(corpus_texts)

    def encode_documents(self, texts: list[str]) -> list[dict]:
        return self._enc().encode_documents(texts)

    def encode_query(self, text: str) -> dict:
        # pinecone-text exposes encode_queries (plural, list-in/list-out); we
        # keep a single-query convenience method for the retriever.
        return self._enc().encode_queries([text])[0]

    def dump(self, path: str) -> None:
        self._enc().dump(path)

    def load(self, path: str) -> None:
        self._enc().load(path)
