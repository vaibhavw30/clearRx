from __future__ import annotations

import json

import pytest

from app.rag.corpus import CorpusError, load_corpus


def _write(dirpath, name, obj):
    p = dirpath / name
    p.write_text(json.dumps(obj))
    return p


def _doc(id_):
    return dict(
        id=id_, drug_a="warfarin", drug_b="ibuprofen",
        drug_a_aliases=[], drug_b_aliases=[],
        drug_class_a="anticoagulant", drug_class_b="nsaid",
        severity="high", sections={"summary": "x"}, evidence=[],
    )


def test_loads_and_sorts(tmp_path):
    _write(tmp_path, "b.json", _doc("int_b"))
    _write(tmp_path, "a.json", _doc("int_a"))
    docs = load_corpus(str(tmp_path))
    assert [d.id for d in docs] == ["int_a", "int_b"]


def test_rejects_duplicate_ids(tmp_path):
    _write(tmp_path, "one.json", _doc("int_dup"))
    _write(tmp_path, "two.json", _doc("int_dup"))
    with pytest.raises(CorpusError):
        load_corpus(str(tmp_path))


def test_rejects_malformed(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(CorpusError):
        load_corpus(str(tmp_path))
