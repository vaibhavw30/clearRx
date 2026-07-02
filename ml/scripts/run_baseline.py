from __future__ import annotations

import re

from app.config import get_settings
from app.eval.baseline import KeywordBaseline
from app.eval.dataset import load_queries
from app.eval.judge import LLMJudge
from app.eval.runner import EvalRunner
from app.rag.corpus import load_corpus


def _offline_judge(prompt: str) -> str:
    """Phase 0 has no LLM. Grade every fact False so fact_coverage is a
    conservative floor for the baseline. Replaced by a real judge in Phase 1+.
    Returns one boolean per numbered fact in the prompt's FACTS section."""
    facts_section = prompt.split("FACTS:", 1)[-1]
    n = len(re.findall(r"(?m)^\d+\.\s", facts_section))
    return "[" + ", ".join("false" for _ in range(n)) + "]"


def main() -> None:
    settings = get_settings()
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    runner = EvalRunner(
        KeywordBaseline(docs), LLMJudge(_offline_judge, max_retries=2), k=5
    )
    report = runner.run(queries)
    json_path, md_path = report.write("eval/reports", run_id="baseline")
    print(report.to_markdown())
    print(f"wrote {json_path} and {md_path}")
    print(f"(embedding model configured: {settings.embedding_model})")


if __name__ == "__main__":
    main()
