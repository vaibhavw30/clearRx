from __future__ import annotations

import json

import pytest

from app.eval.dataset import DatasetError, EvalQuery, load_queries


def _q(id_="q001", **over):
    base = dict(
        id=id_, query="Can I take ibuprofen with warfarin?",
        query_type="interaction", expected_doc_ids=["int_warfarin_ibuprofen"],
        expected_retrieval_topics=["NSAID anticoagulant"],
        expected_answer_facts=["Increased bleeding risk"],
        must_not_say=["safe to combine"], severity="high",
    )
    base.update(over)
    return base


def test_loads_queries(tmp_path):
    p = tmp_path / "queries.json"
    p.write_text(json.dumps({"queries": [_q(), _q("q002")]}))
    qs = load_queries(str(p))
    assert len(qs) == 2 and isinstance(qs[0], EvalQuery)


def test_rejects_bad_query_type(tmp_path):
    p = tmp_path / "queries.json"
    p.write_text(json.dumps({"queries": [_q(query_type="banana")]}))
    with pytest.raises(DatasetError):
        load_queries(str(p))


def test_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "queries.json"
    p.write_text(json.dumps({"queries": [_q("dup"), _q("dup")]}))
    with pytest.raises(DatasetError):
        load_queries(str(p))


def test_no_interaction_query_type_is_accepted(tmp_path):
    from app.eval.dataset import load_queries
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"queries": [{
        "id": "n001",
        "query": "Can I take amoxicillin with acetaminophen?",
        "query_type": "no_interaction",
        "expected_doc_ids": [],
        "expected_retrieval_topics": [],
        "expected_answer_facts": ["No clinically significant interaction"],
        "must_not_say": ["dangerous interaction", "avoid this combination"],
        "severity": "low",
    }]}))
    queries = load_queries(str(p))
    assert queries[0].query_type == "no_interaction"
    assert queries[0].expected_doc_ids == []
