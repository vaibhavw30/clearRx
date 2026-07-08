from __future__ import annotations

from app.eval.aggregate import distinct_doc_ids, mean
from app.eval.metrics import (
    ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, retrieval_coverage,
)

_METRICS = ["retrieval_coverage", "precision_at_k", "recall_at_k", "mrr", "ndcg"]


def evaluate_retrieval(retriever, queries: list, k: int) -> dict:
    cov, prec, rec, mrr, ndcg = [], [], [], [], []
    gradable = 0
    for q in queries:
        chunks = retriever.retrieve(q.query, k)
        ids = distinct_doc_ids(chunks)
        relevant = set(q.expected_doc_ids)
        if q.expected_retrieval_topics:
            cov.append(retrieval_coverage(q.expected_retrieval_topics, [c.text for c in chunks]))
        if relevant:
            gradable += 1
            prec.append(precision_at_k(ids, relevant, k))
            rec.append(recall_at_k(ids, relevant, k))
            mrr.append(reciprocal_rank(ids, relevant))
            ndcg.append(ndcg_at_k(ids, relevant, k))
    return {
        "retrieval_coverage": mean(cov),
        "precision_at_k": mean(prec),
        "recall_at_k": mean(rec),
        "mrr": mean(mrr),
        "ndcg": mean(ndcg),
        "n_queries": float(len(queries)),
        "n_retrieval_gradable": float(gradable),
    }


def compare_strategies(results: dict) -> list[dict]:
    rows = []
    for name, agg in results.items():
        row = {"strategy": name}
        for m in _METRICS:
            row[m] = agg[m]
        rows.append(row)
    return rows


def pick_winner(results: dict, metric: str = "ndcg", tiebreak: str = "recall_at_k") -> str:
    return max(results, key=lambda name: (results[name][metric], results[name][tiebreak]))


def to_markdown(rows: list[dict], winner: str) -> str:
    header = "| strategy | " + " | ".join(_METRICS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in _METRICS) + " |"
    lines = [header, sep]
    for r in rows:
        cells = " | ".join(f"{r[m]:.4f}" for m in _METRICS)
        lines.append(f"| {r['strategy']} | {cells} |")
    lines.append("")
    lines.append(f"**Winner (nDCG, recall tie-break): {winner}**")
    return "\n".join(lines) + "\n"
