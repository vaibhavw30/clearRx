from __future__ import annotations

from app.eval.dataset import EvalQuery
from app.eval.retrieval_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner, to_markdown,
)
from app.rag.models import Chunk


class OneDocRetriever:
    def __init__(self, doc_id):
        self.doc_id = doc_id
    def retrieve(self, query, k):
        return [Chunk(text="nsaid anticoagulant bleeding", source_doc_id=self.doc_id,
                      section="document", chunk_index=0)]


def _queries():
    return [
        EvalQuery(id="q1", query="warfarin ibuprofen", query_type="interaction",
                  expected_doc_ids=["int_warfarin_ibuprofen"],
                  expected_retrieval_topics=["nsaid anticoagulant"],
                  expected_answer_facts=["bleeding"], must_not_say=[], severity="high"),
        EvalQuery(id="n1", query="amox acetaminophen", query_type="no_interaction",
                  expected_doc_ids=[], expected_retrieval_topics=[],
                  expected_answer_facts=["no interaction"], must_not_say=[], severity="low"),
    ]


def test_evaluate_retrieval_gates_on_gradability():
    agg = evaluate_retrieval(OneDocRetriever("int_warfarin_ibuprofen"), _queries(), k=5)
    assert agg["n_queries"] == 2
    assert agg["n_retrieval_gradable"] == 1
    assert agg["recall_at_k"] == 1.0
    assert agg["retrieval_coverage"] == 1.0


def test_pick_winner_ndcg_recall_tiebreak():
    results = {"a": {"ndcg": 0.6, "recall_at_k": 0.5},
               "b": {"ndcg": 0.72, "recall_at_k": 0.66},
               "c": {"ndcg": 0.72, "recall_at_k": 0.61}}
    assert pick_winner(results) == "b"


def test_compare_and_markdown():
    results = {"dense": {"retrieval_coverage": 0.5, "precision_at_k": 0.2,
                         "recall_at_k": 0.5, "mrr": 0.4, "ndcg": 0.45}}
    rows = compare_strategies(results)
    assert rows[0]["strategy"] == "dense" and rows[0]["ndcg"] == 0.45
    md = to_markdown(rows, "dense")
    assert "| strategy |" in md and "dense" in md
