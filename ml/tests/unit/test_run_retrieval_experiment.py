from __future__ import annotations

from scripts.run_retrieval_experiment import pick_best_alpha


def test_pick_best_alpha_by_ndcg_recall_tiebreak():
    sweep = {
        0.0: {"ndcg": 0.50, "recall_at_k": 0.40},
        0.5: {"ndcg": 0.72, "recall_at_k": 0.66},
        1.0: {"ndcg": 0.72, "recall_at_k": 0.55},
    }
    assert pick_best_alpha(sweep) == 0.5  # ties on ndcg (0.72) broken by recall
