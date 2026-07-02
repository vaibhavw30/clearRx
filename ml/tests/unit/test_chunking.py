from __future__ import annotations

from app.rag.chunking import FixedSizeChunker, chunk_metadata
from app.rag.models import Monograph


def _mono():
    words = " ".join(f"w{i}" for i in range(20))
    return Monograph(
        id="d1", drug_a="warfarin", drug_b="ibuprofen",
        drug_a_aliases=["coumadin"], drug_b_aliases=[],
        drug_class_a="anticoagulant", drug_class_b="nsaid",
        severity="high", sections={"summary": words, "mechanism": "short text"},
    )


def test_windows_have_overlap_and_indices():
    chunks = FixedSizeChunker(chunk_size=8, overlap=2).chunk(_mono())
    summary = [c for c in chunks if c.section == "summary"]
    # 20 words, window 8, step 6 -> windows at 0-7, 6-13, 12-19 (the third
    # already reaches the end) => 3 chunks
    assert len(summary) == 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # overlap: last 2 words of window 0 equal first 2 words of window 1
    assert summary[0].text.split()[-2:] == summary[1].text.split()[:2]


def test_metadata_propagates():
    md = chunk_metadata(_mono())
    assert md["severity"] == "high"
    assert "coumadin" in md["drugs_mentioned"]
    assert md["drug_class"] == ["anticoagulant", "nsaid"]


def test_short_section_is_single_chunk():
    chunks = FixedSizeChunker(chunk_size=8, overlap=2).chunk(_mono())
    mech = [c for c in chunks if c.section == "mechanism"]
    assert len(mech) == 1 and mech[0].text == "short text"
