from __future__ import annotations

import json
import os
import time
from typing import Callable

from pydantic import BaseModel

from app.eval.aggregate import distinct_doc_ids, mean
from app.eval.dataset import EvalQuery
from app.eval.judge import Judge
from app.eval.metrics import (
    ndcg_at_k, percentile, precision_at_k, recall_at_k,
    reciprocal_rank, retrieval_coverage,
)
from app.rag.pipeline import RagPipeline


def _mean_where(rows: list[dict], key: str, gate: str) -> float:
    vals = [r[key] for r in rows if r[gate]]
    return sum(vals) / len(vals) if vals else 0.0


class EvalReport(BaseModel):
    aggregate: dict
    per_query: list[dict]

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "| --- | --- |"]
        for key, val in self.aggregate.items():
            lines.append(f"| {key} | {round(val, 4)} |")
        return "\n".join(lines) + "\n"

    def write(self, reports_dir: str, run_id: str) -> tuple[str, str]:
        os.makedirs(reports_dir, exist_ok=True)
        json_path = os.path.join(reports_dir, f"{run_id}.json")
        md_path = os.path.join(reports_dir, f"{run_id}.md")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.model_dump(), fh, indent=2)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(self.to_markdown())
        return json_path, md_path


class EvalRunner:
    def __init__(
        self,
        pipeline: RagPipeline,
        judge: Judge,
        k: int = 5,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.pipeline = pipeline
        self.judge = judge
        self.k = k
        self.clock = clock

    def run(self, queries: list[EvalQuery]) -> EvalReport:
        rows: list[dict] = []
        latencies: list[float] = []
        for q in queries:
            start = self.clock()
            chunks = self.pipeline.retrieve(q.query, self.k)
            answer = self.pipeline.generate(q.query, chunks)
            latency_ms = (self.clock() - start) * 1000.0
            latencies.append(latency_ms)

            retrieved_ids = distinct_doc_ids(chunks)
            relevant = set(q.expected_doc_ids)
            facts = self.judge.score_facts(answer, q.expected_answer_facts)
            forbidden = self.judge.check_forbidden(answer, q.must_not_say)
            rows.append(
                {
                    "id": q.id,
                    "retrieval_coverage": retrieval_coverage(
                        q.expected_retrieval_topics, [c.text for c in chunks]
                    ),
                    "precision_at_k": precision_at_k(retrieved_ids, relevant, self.k),
                    "recall_at_k": recall_at_k(retrieved_ids, relevant, self.k),
                    "mrr": reciprocal_rank(retrieved_ids, relevant),
                    "ndcg": ndcg_at_k(retrieved_ids, relevant, self.k),
                    "fact_coverage": mean([1.0 if f else 0.0 for f in facts]),
                    "forbidden_violations": sum(1 for v in forbidden if v),
                    "latency_ms": latency_ms,
                    "answer": answer,
                    "retrieval_gradable": bool(relevant),
                    "coverage_gradable": bool(q.expected_retrieval_topics),
                }
            )

        aggregate = {
            "retrieval_coverage": _mean_where(rows, "retrieval_coverage", "coverage_gradable"),
            "precision_at_k": _mean_where(rows, "precision_at_k", "retrieval_gradable"),
            "recall_at_k": _mean_where(rows, "recall_at_k", "retrieval_gradable"),
            "mrr": _mean_where(rows, "mrr", "retrieval_gradable"),
            "ndcg": _mean_where(rows, "ndcg", "retrieval_gradable"),
            "fact_coverage": mean([r["fact_coverage"] for r in rows]),
            "forbidden_violations": float(sum(r["forbidden_violations"] for r in rows)),
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
            "n_queries": float(len(rows)),
            "n_retrieval_gradable": float(sum(1 for r in rows if r["retrieval_gradable"])),
        }
        return EvalReport(aggregate=aggregate, per_query=rows)
