from __future__ import annotations

import json
import os

from app.config import get_settings
from app.eval.dataset import load_queries
from app.eval.retrieval_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner, to_markdown,
)
from app.ingest.build_index import build_records
from app.rag.chunking import build_chunkers
from app.rag.embeddings import BGEEmbedder
from app.rag.pipeline import DenseRagPipeline
from app.rag.vectorstore import PineconeStore


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
    md = to_markdown(rows, winner)
    with open(os.path.join(settings.reports_dir, "chunking.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
