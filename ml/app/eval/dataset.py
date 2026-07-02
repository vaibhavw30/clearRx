from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError, field_validator

_TYPES = {"interaction", "dosage", "contraindication"}


class DatasetError(Exception):
    pass


class EvalQuery(BaseModel):
    id: str
    query: str
    query_type: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_retrieval_topics: list[str] = Field(default_factory=list)
    expected_answer_facts: list[str] = Field(default_factory=list)
    must_not_say: list[str] = Field(default_factory=list)
    severity: str

    @field_validator("query_type")
    @classmethod
    def _type_ok(cls, v: str) -> str:
        if v not in _TYPES:
            raise ValueError(f"query_type must be one of {sorted(_TYPES)}")
        return v


def load_queries(path: str) -> list[EvalQuery]:
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise DatasetError(f"{path}: unreadable: {exc}") from exc
    items = raw.get("queries") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        raise DatasetError(f"{path}: expected top-level 'queries' list")
    out: list[EvalQuery] = []
    seen: set[str] = set()
    for item in items:
        try:
            q = EvalQuery(**item)
        except (ValidationError, TypeError) as exc:
            raise DatasetError(f"{path}: invalid query: {exc}") from exc
        if q.id in seen:
            raise DatasetError(f"{path}: duplicate query id {q.id}")
        seen.add(q.id)
        out.append(q)
    return out
