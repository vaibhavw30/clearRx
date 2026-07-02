from __future__ import annotations

import json

from app.eval.dataset import EvalQuery
from app.eval.runner import EvalRunner
from app.rag.models import Chunk


class FakePipeline:
    def retrieve(self, query, k):
        return [
            Chunk(
                text="increased bleeding risk nsaid anticoagulant",
                source_doc_id="int_warfarin_ibuprofen",
                section="summary",
                chunk_index=0,
            )
        ]

    def generate(self, query, chunks):
        return "There is an increased bleeding risk."


class FakeJudge:
    def score_facts(self, answer, facts):
        return [True for _ in facts]

    def check_forbidden(self, answer, must_not_say):
        return [False for _ in must_not_say]


def _query():
    return EvalQuery(
        id="q001", query="ibuprofen warfarin", query_type="interaction",
        expected_doc_ids=["int_warfarin_ibuprofen"],
        expected_retrieval_topics=["nsaid anticoagulant"],
        expected_answer_facts=["Increased bleeding risk"],
        must_not_say=["safe to combine"], severity="high",
    )


def test_run_computes_perfect_scores():
    ticks = iter([0.0, 0.010])  # start, end for the single query
    runner = EvalRunner(FakePipeline(), FakeJudge(), k=5, clock=lambda: next(ticks))
    report = runner.run([_query()])
    agg = report.aggregate
    assert agg["retrieval_coverage"] == 1.0
    assert agg["precision_at_k"] == 1.0  # the one retrieved chunk is relevant
    assert agg["recall_at_k"] == 1.0
    assert agg["mrr"] == 1.0
    assert agg["fact_coverage"] == 1.0
    assert agg["forbidden_violations"] == 0
    assert agg["latency_ms_p50"] == 10.0


def test_write_report_creates_files(tmp_path):
    runner = EvalRunner(FakePipeline(), FakeJudge(), k=5, clock=lambda: 0.0)
    report = runner.run([_query()])
    json_path, md_path = report.write(str(tmp_path), run_id="baseline")
    assert json.loads(open(json_path).read())["aggregate"]["mrr"] == 1.0
    assert "| metric |" in open(md_path).read().lower()
