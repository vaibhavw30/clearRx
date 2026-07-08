from __future__ import annotations

import json
import os

from app.config import get_settings
from app.eval.aggregate import distinct_doc_ids, mean
from app.eval.dataset import load_queries
from app.eval.metrics import (
    ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, retrieval_coverage,
)
from app.ingest.build_index import build_records
from app.rag.chunking import build_chunkers
from app.rag.embeddings import BGEEmbedder
from app.rag.pipeline import DenseRagPipeline
from app.rag.vectorstore import PineconeStore

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


def _to_markdown(rows: list[dict], winner: str) -> str:
    header = "| strategy | " + " | ".join(_METRICS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in _METRICS) + " |"
    lines = [header, sep]
    for r in rows:
        cells = " | ".join(f"{r[m]:.4f}" for m in _METRICS)
        lines.append(f"| {r['strategy']} | {cells} |")
    lines.append("")
    lines.append(f"**Winner (nDCG, recall tie-break): {winner}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    queries = load_queries(os.path.join(settings.eval_dir, "queries.json"))
    from app.rag.corpus import load_corpus

    docs = load_corpus(settings.corpus_dir)
    embedder = BGEEmbedder(settings.embedding_model, settings.embedding_dim)
    store = PineconeStore(settings.pinecone_api_key, settings.pinecone_index)
    store.ensure_index(settings.embedding_dim, settings.pinecone_metric,
                       settings.pinecone_cloud, settings.pinecone_region)

    results: dict = {}
    for chunker in build_chunkers(settings, embedder):
        namespace = f"chunk_{chunker.name}"
        records = build_records(docs, chunker, embedder)
        store.upsert(records, namespace=namespace)
        pipeline = DenseRagPipeline(embedder, store, llm=None, namespace=namespace)
        results[chunker.name] = evaluate_retrieval(pipeline, queries, k=5)
        print(f"{chunker.name}: {len(records)} chunks -> {results[chunker.name]}")

    rows = compare_strategies(results)
    winner = pick_winner(results)
    os.makedirs(settings.reports_dir, exist_ok=True)
    with open(os.path.join(settings.reports_dir, "chunking.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": results, "winner": winner}, fh, indent=2)
    md = _to_markdown(rows, winner)
    with open(os.path.join(settings.reports_dir, "chunking.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
