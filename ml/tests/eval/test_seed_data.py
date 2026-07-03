from __future__ import annotations

from app.eval.baseline import KeywordBaseline
from app.eval.dataset import load_queries
from app.eval.runner import EvalRunner
from app.rag.corpus import load_corpus


class _AllTrueJudge:
    def score_facts(self, answer, facts):
        return [True for _ in facts]

    def check_forbidden(self, answer, must_not_say):
        return [phrase.lower() in answer.lower() for phrase in must_not_say]


def test_seed_corpus_and_queries_load():
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    doc_ids = {d.id for d in docs}
    assert len(docs) >= 25
    for q in queries:
        for did in q.expected_doc_ids:
            assert did in doc_ids, f"{q.id} references unknown doc {did}"


def test_every_monograph_is_complete_and_cited():
    docs = load_corpus("data/corpus")
    required = {"summary", "mechanism", "clinical_effects", "management", "monitoring"}
    for d in docs:
        assert required.issubset(d.sections), f"{d.id} missing sections"
        assert all(d.sections[s].strip() for s in required), f"{d.id} has an empty section"
        assert d.evidence, f"{d.id} has no evidence"
        assert all(e.url.startswith("http") for e in d.evidence), f"{d.id} evidence missing url"


def test_baseline_runs_end_to_end_offline():
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    runner = EvalRunner(KeywordBaseline(docs), _AllTrueJudge(), k=5, clock=lambda: 0.0)
    report = runner.run(queries)
    assert 0.0 <= report.aggregate["recall_at_k"] <= 1.0
    # the keyword baseline should retrieve the right doc for most queries
    assert report.aggregate["recall_at_k"] > 0.4


def test_query_set_size_and_shape():
    queries = load_queries("data/eval/queries.json")
    assert len(queries) >= 40
    by_type = {}
    for q in queries:
        by_type.setdefault(q.query_type, 0)
        by_type[q.query_type] += 1
    assert by_type.get("no_interaction", 0) >= 8, "need negative queries for precision"
    # every non-negative query names at least one gold doc
    for q in queries:
        if q.query_type != "no_interaction":
            assert q.expected_doc_ids, f"{q.id} has no gold doc"


def test_human_labels_align_with_queries():
    import json
    queries = {q.id: q for q in load_queries("data/eval/queries.json")}
    labels = json.load(open("data/eval/human_labels.json"))["labels"]
    assert len(labels) >= 15
    for lab in labels:
        q = queries[lab["query_id"]]
        assert len(lab["human_fact_labels"]) == len(q.expected_answer_facts), (
            f"{lab['query_id']} label length != expected_answer_facts length"
        )
