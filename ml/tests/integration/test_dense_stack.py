from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings


@pytest.mark.integration
def test_bge_real_dimension_and_norm():
    from app.rag.embeddings import BGEEmbedder

    s = get_settings()
    emb = BGEEmbedder(s.embedding_model, s.embedding_dim)
    out = emb.embed(["warfarin ibuprofen bleeding risk"])
    assert out.shape == (1, s.embedding_dim)          # 1024
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-3)
    q = emb.embed_query("can I take advil with coumadin")
    assert q.shape == (s.embedding_dim,)
