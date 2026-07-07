from __future__ import annotations

from app.eval.calibration import calibrate
from app.eval.dataset import EvalQuery


class ScriptedJudge:
    """Returns preset booleans per query so agreement is hand-computable."""

    def __init__(self, mapping):
        self.mapping = mapping

    def score_facts(self, answer, facts):
        return self.mapping[answer]

    def check_forbidden(self, answer, must_not_say):
        return [False for _ in must_not_say]


def _q(qid, facts):
    return EvalQuery(id=qid, query="x", query_type="interaction",
                     expected_doc_ids=["d"], expected_answer_facts=facts,
                     must_not_say=[], severity="high")


def test_calibrate_computes_fact_level_agreement():
    queries = {"q1": _q("q1", ["f1", "f2", "f3"]), "q2": _q("q2", ["f1", "f2"])}
    labels = [
        {"query_id": "q1", "answer": "A1", "human_fact_labels": [True, False, True]},
        {"query_id": "q2", "answer": "A2", "human_fact_labels": [True, True]},
    ]
    judge = ScriptedJudge({"A1": [True, False, False], "A2": [True, True]})
    #   q1: judge [T,F,F] vs human [T,F,T] -> 2/3 agree
    #   q2: judge [T,T]   vs human [T,T]   -> 2/2 agree
    #   total 4/5 = 0.8
    result = calibrate(judge, queries, labels)
    assert result["n_labels"] == 2
    assert result["n_facts"] == 5
    assert abs(result["agreement"] - 0.8) < 1e-9
    per = {p["query_id"]: p for p in result["per_query"]}
    assert per["q1"] == {"query_id": "q1", "matches": 2, "n": 3}
