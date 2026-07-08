from __future__ import annotations

import json
import os

from app.config import get_settings
from app.eval.dataset import load_queries
from app.eval.retrieval_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner, to_markdown,
)
from app.ingest.build_index import build_records
from app.rag.chunking import FixedSizeChunker
from app.rag.corpus import load_corpus
from app.rag.embeddings import BGEEmbedder
from app.rag.hybrid import HybridRerankRetriever
from app.rag.rerank import build_reranker
from app.rag.sparse import BM25SparseEncoder
from app.rag.vectorstore import PineconeStore


def pick_best_alpha(sweep: dict) -> float:
    """The alpha whose hybrid config maximizes nDCG (recall tie-break)."""
    return pick_winner(sweep)


def main() -> None:
    settings = get_settings()
    queries = load_queries(os.path.join(settings.eval_dir, "queries.json"))
    docs = load_corpus(settings.corpus_dir)
    embedder = BGEEmbedder(settings.embedding_model, settings.embedding_dim)
    sparse = BM25SparseEncoder()
    store = PineconeStore(settings.pinecone_api_key, settings.pinecone_index)
    store.ensure_index(settings.embedding_dim, settings.pinecone_metric,
                       settings.pinecone_cloud, settings.pinecone_region)

    # Build the hybrid namespace (dense + BM25 sparse); persist BM25 params.
    records = build_records(
        docs, FixedSizeChunker(chunk_size=512, overlap=0), embedder, sparse_encoder=sparse
    )
    store.upsert(records, namespace=settings.hybrid_namespace)
    sparse.dump(settings.bm25_params_path)
    print(f"indexed {len(records)} hybrid chunks -> namespace '{settings.hybrid_namespace}'")

    # Alpha sweep, retrieval-only (rerank off). alpha=1.0 is dense-only,
    # alpha=0.0 is sparse-only.
    sweep = {}
    for a in settings.hybrid_alphas:
        retriever = HybridRerankRetriever(
            embedder, sparse, store, settings.hybrid_namespace,
            reranker=None, alpha=a, top_k=settings.hybrid_top_k,
        )
        sweep[a] = evaluate_retrieval(retriever, queries, k=5)
        print(f"alpha={a}: {sweep[a]}")
    best_alpha = pick_best_alpha(sweep)

    # Hybrid(best alpha) + rerank on.
    reranker = build_reranker(settings)
    hybrid_rerank = HybridRerankRetriever(
        embedder, sparse, store, settings.hybrid_namespace,
        reranker=reranker, alpha=best_alpha, top_k=settings.hybrid_top_k,
    )

    results = {f"hybrid(alpha={a})": agg for a, agg in sweep.items()}
    results[f"hybrid(alpha={best_alpha})+rerank"] = evaluate_retrieval(hybrid_rerank, queries, k=5)

    rows = compare_strategies(results)
    winner = pick_winner(results)
    os.makedirs(settings.reports_dir, exist_ok=True)
    with open(os.path.join(settings.reports_dir, "retrieval.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": results, "winner": winner, "best_alpha": best_alpha}, fh, indent=2)
    md = to_markdown(rows, winner)
    with open(os.path.join(settings.reports_dir, "retrieval.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
