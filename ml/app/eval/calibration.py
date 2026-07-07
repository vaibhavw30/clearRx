from __future__ import annotations

import json


def load_labels(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return raw["labels"]


def calibrate(judge, queries_by_id: dict, labels: list[dict]) -> dict:
    total = 0
    agree = 0
    per_query: list[dict] = []
    for lab in labels:
        q = queries_by_id[lab["query_id"]]
        pred = judge.score_facts(lab["answer"], q.expected_answer_facts)
        human = lab["human_fact_labels"]
        matches = sum(1 for p, h in zip(pred, human) if bool(p) == bool(h))
        total += len(human)
        agree += matches
        per_query.append({"query_id": lab["query_id"], "matches": matches, "n": len(human)})
    return {
        "agreement": agree / total if total else 0.0,
        "n_facts": total,
        "n_labels": len(labels),
        "per_query": per_query,
    }
