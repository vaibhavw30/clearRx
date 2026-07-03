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


class MultiChunkSameDocPipeline:
    """Real corpora produce several chunks per document, so retrieval returns
    the same doc id repeated. Doc-level ranking metrics must not double-count."""

    def retrieve(self, query, k):
        return [
            Chunk(text="a", source_doc_id="int_warfarin_ibuprofen", section="summary", chunk_index=0),
            Chunk(text="b", source_doc_id="int_warfarin_ibuprofen", section="mechanism", chunk_index=1),
            Chunk(text="c", source_doc_id="int_warfarin_ibuprofen", section="management", chunk_index=2),
        ]

    def generate(self, query, chunks):
        return "answer"


def test_duplicate_doc_chunks_do_not_inflate_ranking_metrics():
    runner = EvalRunner(MultiChunkSameDocPipeline(), FakeJudge(), k=5, clock=lambda: 0.0)
    report = runner.run([_query()])
    agg = report.aggregate
    assert agg["ndcg"] <= 1.0
    assert agg["ndcg"] == 1.0            # the one relevant doc is ranked first
    assert agg["precision_at_k"] == 1.0  # one distinct doc, and it is relevant
    assert agg["recall_at_k"] == 1.0
    assert agg["mrr"] == 1.0


def _negative_query():
    return EvalQuery(
        id="n001", query="amoxicillin acetaminophen", query_type="no_interaction",
        expected_doc_ids=[], expected_retrieval_topics=[],
        expected_answer_facts=["No clinically significant interaction"],
        must_not_say=["dangerous interaction"], severity="low",
    )


def test_negative_query_does_not_drag_ranking_metrics():
    # Pipeline always returns the warfarin doc; for the positive query that is
    # relevant (recall 1.0). The negative query has no gold doc and must be
    # excluded from ranking aggregates rather than counted as a 0.0.
    runner = EvalRunner(MultiChunkSameDocPipeline(), FakeJudge(), k=5, clock=lambda: 0.0)
    report = runner.run([_query(), _negative_query()])
    agg = report.aggregate
    assert agg["n_queries"] == 2.0
    assert agg["n_retrieval_gradable"] == 1.0
    assert agg["recall_at_k"] == 1.0   # averaged over the 1 gradable query, not 0.5
    assert agg["mrr"] == 1.0
    # per-query rows expose gradability
    rows = {r["id"]: r for r in report.per_query}
    assert rows["q001"]["retrieval_gradable"] is True
    assert rows["n001"]["retrieval_gradable"] is False


def test_write_report_creates_files(tmp_path):
    runner = EvalRunner(FakePipeline(), FakeJudge(), k=5, clock=lambda: 0.0)
    report = runner.run([_query()])
    json_path, md_path = report.write(str(tmp_path), run_id="baseline")
    assert json.loads(open(json_path).read())["aggregate"]["mrr"] == 1.0
    assert "| metric |" in open(md_path).read().lower()
