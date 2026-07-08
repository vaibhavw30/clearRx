from __future__ import annotations

from app.rag.sparse import BM25SparseEncoder


class FakeBM25:
    def __init__(self):
        self.fitted = None
        self.dumped = None
        self.loaded = None
    def fit(self, corpus):
        self.fitted = list(corpus)
    def encode_documents(self, texts):
        return [{"indices": [1], "values": [0.5]} for _ in texts]
    def encode_query(self, text):
        return {"indices": [1], "values": [0.9]}
    def dump(self, path):
        self.dumped = path
    def load(self, path):
        self.loaded = path


def test_fit_and_encode_delegate_to_encoder():
    fake = FakeBM25()
    enc = BM25SparseEncoder(encoder=fake)
    enc.fit(["a b", "c d"])
    assert fake.fitted == ["a b", "c d"]
    docs = enc.encode_documents(["a b", "c d"])
    assert len(docs) == 2 and docs[0] == {"indices": [1], "values": [0.5]}
    assert enc.encode_query("a") == {"indices": [1], "values": [0.9]}


def test_dump_load_delegate(tmp_path):
    fake = FakeBM25()
    enc = BM25SparseEncoder(encoder=fake)
    p = str(tmp_path / "bm25.json")
    enc.dump(p)
    enc.load(p)
    assert fake.dumped == p and fake.loaded == p
