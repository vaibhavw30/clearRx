from __future__ import annotations

from app.eval.dataset import EvalQuery
from app.rag.models import Chunk
from scripts.run_chunking_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner,
)


class OneDocRetriever:
    """Always returns chunks from the given doc id, so retrieval metrics are
    fully determined by whether that doc is the gold doc."""
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
        EvalQuery(id="n1", query="amoxicillin acetaminophen", query_type="no_interaction",
                  expected_doc_ids=[], expected_retrieval_topics=[],
                  expected_answer_facts=["no interaction"], must_not_say=[], severity="low"),
    ]


def test_evaluate_retrieval_gates_on_gradability():
    agg = evaluate_retrieval(OneDocRetriever("int_warfarin_ibuprofen"), _queries(), k=5)
    assert agg["n_queries"] == 2
    assert agg["n_retrieval_gradable"] == 1        # only the positive query has a gold doc
    assert agg["recall_at_k"] == 1.0               # averaged over the 1 gradable query
    assert agg["mrr"] == 1.0
    assert agg["retrieval_coverage"] == 1.0        # "nsaid anticoagulant" is in the chunk text


def test_pick_winner_by_ndcg_with_recall_tiebreak():
    results = {
        "fixed": {"ndcg": 0.60, "recall_at_k": 0.50},
        "recursive": {"ndcg": 0.72, "recall_at_k": 0.66},
        "semantic": {"ndcg": 0.72, "recall_at_k": 0.61},
    }
    assert pick_winner(results) == "recursive"     # ties on ndcg (0.72) broken by recall


def test_compare_strategies_emits_one_row_per_strategy():
    results = {
        "fixed": {"retrieval_coverage": 0.5, "precision_at_k": 0.2, "recall_at_k": 0.5,
                  "mrr": 0.4, "ndcg": 0.45},
        "recursive": {"retrieval_coverage": 0.6, "precision_at_k": 0.25, "recall_at_k": 0.66,
                      "mrr": 0.5, "ndcg": 0.55},
    }
    rows = compare_strategies(results)
    assert {r["strategy"] for r in rows} == {"fixed", "recursive"}
    fixed = next(r for r in rows if r["strategy"] == "fixed")
    assert fixed["ndcg"] == 0.45
