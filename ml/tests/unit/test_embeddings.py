from __future__ import annotations

import numpy as np

from app.rag.embeddings import BGEEmbedder, QUERY_INSTRUCTION


class FakeST:
    """Stand-in for SentenceTransformer that records inputs and returns
    fixed, un-normalized vectors so we can assert the wrapper normalizes."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True):
        self.calls.append({"texts": list(texts), "normalize": normalize_embeddings})
        base = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        arr = base[: len(texts)]
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def test_embed_documents_are_normalized_and_shaped():
    fake = FakeST()
    emb = BGEEmbedder("bge", dimension=2, model=fake)
    out = emb.embed(["doc one", "doc two"])
    assert out.shape == (2, 2)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert fake.calls[0]["normalize"] is True
    assert fake.calls[0]["texts"] == ["doc one", "doc two"]  # no instruction on docs


def test_embed_query_prepends_instruction_and_is_1d():
    fake = FakeST()
    emb = BGEEmbedder("bge", dimension=2, model=fake)
    vec = emb.embed_query("ibuprofen warfarin")
    assert vec.shape == (2,)
    assert fake.calls[0]["texts"] == [QUERY_INSTRUCTION + "ibuprofen warfarin"]


def test_dimension_attribute():
    assert BGEEmbedder("bge", dimension=1024, model=FakeST()).dimension == 1024
