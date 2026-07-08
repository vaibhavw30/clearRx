from __future__ import annotations

import numpy as np

from app.rag.chunking import FixedSizeChunker, RecursiveChunker, SemanticChunker, chunk_metadata
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


class FakeSplitter:
    """Stands in for LangChain's RecursiveCharacterTextSplitter."""
    def __init__(self):
        self.seen = None
    def split_text(self, text):
        self.seen = text
        return ["piece one", "piece two"]


def _doc():
    return Monograph(
        id="int_a_b", drug_a="a", drug_b="b", drug_class_a="x", drug_class_b="y",
        severity="high",
        sections={"summary": "a and b interact", "management": "avoid combining"},
    )


def test_recursive_chunker_uses_splitter_over_whole_document():
    fake = FakeSplitter()
    chunker = RecursiveChunker(splitter=fake)
    chunks = chunker.chunk(_doc())
    assert chunker.name == "recursive"
    assert [c.text for c in chunks] == ["piece one", "piece two"]
    # whole-document input: both section texts are present in what the splitter saw
    assert "a and b interact" in fake.seen and "avoid combining" in fake.seen
    assert [c.chunk_index for c in chunks] == [0, 1]
    for c in chunks:
        assert c.source_doc_id == "int_a_b"
        assert c.section == "document"
        assert c.metadata["severity"] == "high"
        assert "a" in c.metadata["drugs_mentioned"]


def test_recursive_chunker_drops_empty_pieces():
    class Emptyish:
        def split_text(self, text):
            return ["real chunk", "   ", ""]
    chunks = RecursiveChunker(splitter=Emptyish()).chunk(_doc())
    assert [c.text for c in chunks] == ["real chunk"]


class ClusterEmbedder:
    """Maps sentences to one of two orthogonal unit vectors based on a keyword,
    so a semantic boundary falls exactly between the two groups."""
    dimension = 2

    def embed(self, texts):
        rows = []
        for t in texts:
            rows.append([1.0, 0.0] if "alpha" in t else [0.0, 1.0])
        return np.array(rows, dtype=np.float32)

    def embed_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def _two_topic_doc():
    return Monograph(
        id="int_a_b", drug_a="a", drug_b="b", drug_class_a="x", drug_class_b="y",
        severity="moderate",
        sections={
            "summary": "First alpha sentence here. Second alpha sentence here.",
            "mechanism": "First beta sentence here. Second beta sentence here.",
        },
    )


def test_semantic_chunker_breaks_at_topic_boundary():
    chunker = SemanticChunker(ClusterEmbedder(), threshold_percentile=85.0)
    chunks = chunker.chunk(_two_topic_doc())
    assert chunker.name == "semantic"
    # the two alpha sentences group together; the two beta sentences group together
    assert len(chunks) == 2
    assert "alpha" in chunks[0].text and "beta" not in chunks[0].text
    assert "beta" in chunks[1].text and "alpha" not in chunks[1].text
    for c in chunks:
        assert c.source_doc_id == "int_a_b"
        assert c.section == "document"
        assert c.metadata["severity"] == "moderate"


def test_semantic_chunker_single_sentence_is_one_chunk():
    doc = Monograph(
        id="int_c_d", drug_a="c", drug_b="d", drug_class_a="x", drug_class_b="y",
        severity="low", sections={"summary": "Only one sentence"},
    )
    chunks = SemanticChunker(ClusterEmbedder()).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence"
