from __future__ import annotations

from app.eval.baseline import KeywordBaseline
from app.rag.models import Monograph


def _docs():
    return [
        Monograph(
            id="int_warfarin_ibuprofen", drug_a="warfarin", drug_b="ibuprofen",
            drug_class_a="anticoagulant", drug_class_b="nsaid", severity="high",
            sections={"summary": "warfarin ibuprofen increased bleeding risk monitor INR"},
        ),
        Monograph(
            id="int_metformin_contrast", drug_a="metformin", drug_b="contrast",
            drug_class_a="antidiabetic", drug_class_b="contrast", severity="moderate",
            sections={"summary": "metformin contrast lactic acidosis kidney"},
        ),
    ]


def test_retrieve_ranks_relevant_doc_first():
    bl = KeywordBaseline(_docs())
    chunks = bl.retrieve("can I take ibuprofen with warfarin bleeding", k=1)
    assert len(chunks) == 1
    assert chunks[0].source_doc_id == "int_warfarin_ibuprofen"


def test_generate_is_deterministic_and_grounded():
    bl = KeywordBaseline(_docs())
    chunks = bl.retrieve("ibuprofen warfarin", k=1)
    ans = bl.generate("ibuprofen warfarin", chunks)
    assert "bleeding" in ans.lower()
    assert bl.generate("ibuprofen warfarin", chunks) == ans  # deterministic


def test_retrieve_returns_nothing_when_no_overlap():
    bl = KeywordBaseline(_docs())
    assert bl.retrieve("zzz qqq unrelated", k=3) == []
