from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.models import Chunk, Evidence, Monograph


def _mono(**over):
    base = dict(
        id="int_warfarin_ibuprofen",
        drug_a="warfarin",
        drug_b="ibuprofen",
        drug_a_aliases=["Coumadin"],
        drug_b_aliases=["Advil", "Motrin"],
        drug_class_a="anticoagulant",
        drug_class_b="nsaid",
        severity="high",
        sections={"summary": "Increased bleeding risk."},
        evidence=[Evidence(citation="Ann Pharmacother 2004", url="https://example.org/1")],
    )
    base.update(over)
    return Monograph(**base)


def test_valid_monograph_and_drug_names():
    m = _mono()
    names = m.all_drug_names()
    assert "warfarin" in names and "coumadin" in names and "advil" in names
    assert all(n == n.lower() for n in names)


def test_rejects_bad_severity():
    with pytest.raises(ValidationError):
        _mono(severity="critical")


def test_rejects_empty_sections():
    with pytest.raises(ValidationError):
        _mono(sections={})


def test_chunk_defaults_metadata():
    c = Chunk(text="hi", source_doc_id="d1", section="summary", chunk_index=0)
    assert c.metadata == {}
