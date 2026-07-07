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
