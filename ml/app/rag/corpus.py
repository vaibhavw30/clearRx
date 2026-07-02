from __future__ import annotations

import glob
import json
import os

from pydantic import ValidationError

from app.rag.models import Monograph


class CorpusError(Exception):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


def load_corpus(corpus_dir: str) -> list[Monograph]:
    docs: list[Monograph] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise CorpusError(path, f"unreadable: {exc}") from exc
        try:
            doc = Monograph(**raw)
        except ValidationError as exc:
            raise CorpusError(path, f"invalid monograph: {exc}") from exc
        if doc.id in seen:
            raise CorpusError(path, f"duplicate id {doc.id}")
        seen.add(doc.id)
        docs.append(doc)
    return sorted(docs, key=lambda d: d.id)
