from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel


class Record(BaseModel):
    id: str
    values: list[float]
    metadata: dict


class Match(BaseModel):
    id: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    def upsert(self, records: list[Record], namespace: str) -> None: ...
    def query(
        self, dense: list[float], top_k: int, flt: Optional[dict], namespace: str
    ) -> list[Match]: ...


class PineconeStore:
    """Pinecone serverless dense store. Inject `index` (and/or `client`) in
    tests to avoid importing the SDK or hitting the network."""

    def __init__(self, api_key: str, index_name: str, *, client=None, index=None) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self._client = client
        self._index = index

    def _pc(self):
        if self._client is None:
            from pinecone import Pinecone  # lazy

            self._client = Pinecone(api_key=self.api_key)
        return self._client

    def _idx(self):
        if self._index is None:
            self._index = self._pc().Index(self.index_name)
        return self._index

    def ensure_index(self, dimension: int, metric: str, cloud: str, region: str) -> None:
        from pinecone import ServerlessSpec  # lazy

        pc = self._pc()
        existing = {i["name"] for i in pc.list_indexes()}
        if self.index_name not in existing:
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

    def upsert(self, records: list[Record], namespace: str) -> None:
        vectors = [
            {"id": r.id, "values": list(r.values), "metadata": r.metadata} for r in records
        ]
        self._idx().upsert(vectors=vectors, namespace=namespace)

    def query(
        self, dense: list[float], top_k: int, flt: Optional[dict], namespace: str
    ) -> list[Match]:
        res = self._idx().query(
            vector=list(dense),
            top_k=top_k,
            include_metadata=True,
            filter=flt,
            namespace=namespace,
        )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out: list[Match] = []
        for m in matches:
            md = m["metadata"] if isinstance(m, dict) else m.metadata
            mid = m["id"] if isinstance(m, dict) else m.id
            score = m["score"] if isinstance(m, dict) else m.score
            out.append(Match(id=mid, score=float(score), metadata=dict(md)))
        return out
