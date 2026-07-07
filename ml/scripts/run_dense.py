from __future__ import annotations

import json
import os

from app.config import get_settings
from app.eval.calibration import calibrate, load_labels
from app.eval.dataset import load_queries
from app.eval.runner import EvalRunner
from app.rag.embeddings import BGEEmbedder
from app.rag.generator import OllamaClient
from app.rag.judge_clients import build_judge
from app.rag.pipeline import DenseRagPipeline
from app.rag.vectorstore import PineconeStore

_SHARED = ["retrieval_coverage", "precision_at_k", "recall_at_k", "mrr", "ndcg", "fact_coverage"]


def compare(baseline: dict, dense: dict) -> list[dict]:
    rows = []
    for m in _SHARED:
        if m in baseline and m in dense:
            rows.append(
                {"metric": m, "baseline": baseline[m], "dense": dense[m],
                 "delta": dense[m] - baseline[m]}
            )
    return rows


def main() -> None:
    settings = get_settings()
    queries = load_queries(os.path.join(settings.eval_dir, "queries.json"))

    embedder = BGEEmbedder(settings.embedding_model, settings.embedding_dim)
    store = PineconeStore(settings.pinecone_api_key, settings.pinecone_index)
    llm = OllamaClient(settings.ollama_base_url, settings.gen_model)
    pipeline = DenseRagPipeline(embedder, store, llm, namespace=settings.pinecone_namespace)
    judge = build_judge(settings)

    report = EvalRunner(pipeline, judge, k=5).run(queries)
    report.write(settings.reports_dir, run_id="dense")

    labels = load_labels(os.path.join(settings.eval_dir, "human_labels.json"))
    cal = calibrate(judge, {q.id: q for q in queries}, labels)

    baseline_path = os.path.join(settings.reports_dir, "baseline.json")
    print("\n=== dense metrics ===")
    print(report.to_markdown())
    if os.path.exists(baseline_path):
        baseline = json.load(open(baseline_path))["aggregate"]
        print("=== keyword -> dense ===")
        for r in compare(baseline, report.aggregate):
            print(f"{r['metric']:>20}: {r['baseline']:.4f} -> {r['dense']:.4f}  "
                  f"(delta {r['delta']:+.4f})")
    print(f"\n=== judge calibration ===\njudge-vs-human agreement: "
          f"{cal['agreement']:.1%} over {cal['n_facts']} facts / {cal['n_labels']} labels")
    if cal["agreement"] < 0.8:
        print("agreement < 80% -> consider JUDGE_PROVIDER=openai JUDGE_MODEL=gpt-4o-mini")


if __name__ == "__main__":
    main()
