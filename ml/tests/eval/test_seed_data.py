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
    assert len(docs) >= 8
    assert len(queries) >= 12
    # every expected_doc_id in the eval set must exist in the corpus
    for q in queries:
        for did in q.expected_doc_ids:
            assert did in doc_ids, f"{q.id} references unknown doc {did}"


def test_baseline_runs_end_to_end_offline():
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    runner = EvalRunner(KeywordBaseline(docs), _AllTrueJudge(), k=5, clock=lambda: 0.0)
    report = runner.run(queries)
    assert 0.0 <= report.aggregate["recall_at_k"] <= 1.0
    # the keyword baseline should retrieve the right doc for most queries
    assert report.aggregate["recall_at_k"] > 0.4
