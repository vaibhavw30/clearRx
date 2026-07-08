from __future__ import annotations

from app.rag.vectorstore import Match, PineconeStore, Record


class FakeIndex:
    def __init__(self):
        self.upserted = []
        self.queries = []

    def upsert(self, vectors, namespace):
        self.upserted.append({"vectors": vectors, "namespace": namespace})

    def query(self, vector, top_k, include_metadata, filter, namespace):
        self.queries.append(
            {"vector": vector, "top_k": top_k, "include_metadata": include_metadata,
             "filter": filter, "namespace": namespace}
        )
        return {
            "matches": [
                {"id": "int_x::summary::0", "score": 0.91,
                 "metadata": {"chunk_text": "bleeding risk", "source_doc_id": "int_x"}},
            ]
        }


def test_upsert_maps_records_to_pinecone_vectors():
    idx = FakeIndex()
    store = PineconeStore("k", "clearrx", index=idx)
    store.upsert(
        [Record(id="r1", values=[0.1, 0.2], metadata={"chunk_text": "t"})], namespace="curated"
    )
    call = idx.upserted[0]
    assert call["namespace"] == "curated"
    assert call["vectors"][0] == {"id": "r1", "values": [0.1, 0.2], "metadata": {"chunk_text": "t"}}


def test_query_returns_match_objects():
    idx = FakeIndex()
    store = PineconeStore("k", "clearrx", index=idx)
    matches = store.query([0.1, 0.2], top_k=5, flt=None, namespace="curated")
    assert matches == [
        Match(id="int_x::summary::0", score=0.91,
              metadata={"chunk_text": "bleeding risk", "source_doc_id": "int_x"})
    ]
    assert idx.queries[0]["top_k"] == 5
    assert idx.queries[0]["include_metadata"] is True   # store always requests metadata
    assert idx.queries[0]["filter"] is None


class _SparseFakeIndex:
    def __init__(self):
        self.upserted = None
        self.queried = None
    def upsert(self, vectors, namespace):
        self.upserted = (vectors, namespace)
    def query(self, **kw):
        self.queried = kw
        return {"matches": [{"id": "d::s::0", "score": 1.0,
                             "metadata": {"source_doc_id": "d"}}]}


def test_upsert_includes_sparse_when_present():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).upsert(
        [Record(id="x", values=[0.1], sparse_values={"indices": [1], "values": [0.5]}, metadata={})],
        namespace="hybrid")
    assert idx.upserted[0][0]["sparse_values"] == {"indices": [1], "values": [0.5]}


def test_upsert_omits_sparse_when_none():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).upsert(
        [Record(id="x", values=[0.1], metadata={})], namespace="curated")
    assert "sparse_values" not in idx.upserted[0][0]


def test_query_passes_sparse_vector():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).query(
        [0.1], top_k=5, flt=None, namespace="hybrid", sparse={"indices": [1], "values": [0.5]})
    assert idx.queried["sparse_vector"] == {"indices": [1], "values": [0.5]}


def test_query_dense_only_omits_sparse_vector():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).query([0.1], top_k=5, flt=None, namespace="curated")
    assert "sparse_vector" not in idx.queried
