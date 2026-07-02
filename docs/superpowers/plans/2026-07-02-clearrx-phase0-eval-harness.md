# ClearRx Phase 0 — Corpus + Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the evaluation harness and curated corpus foundation for the ClearRx RAG rebuild — before any retrieval backend changes — so every later improvement is measured against an honest baseline.

**Architecture:** Restructure `ml/` into a testable `app/` package. Phase 0 delivers: the corpus data model + loader, a first chunker, the eval dataset model + loader, pure retrieval-metric functions, a pluggable LLM-as-judge, a naive keyword baseline (an honest stand-in for today's name-lookup system, which has no query retrieval at all), and a runner that scores any pipeline and emits diffable json+markdown reports. Everything runs offline with fakes — no Pinecone/Cohere/Ollama in Phase 0.

**Tech Stack:** Python 3.9, pytest, pydantic v2, pydantic-settings, numpy.

## Global Constraints

- Python **3.9.6** (venv at `ml/venv`). Every module starts with `from __future__ import annotations` so `list[str]`/`dict[str, str]` annotations are legal on 3.9.
- pydantic **v2** (already in `ml/requirements.txt` as `pydantic>=2.5.0`).
- All external-service code (Pinecone/Cohere/Ollama) is out of scope for Phase 0; nothing here may require an API key or network.
- Tests use **fakes only**; `pytest` must pass with no environment variables set.
- Retrieval and generation metrics are reported **separately** (spec §7).
- Reports are **deterministic**: any timestamp/run-id is injected by the caller, never read from the clock inside library code (keeps tests reproducible).
- Commit after every task. Run from `ml/` with the venv active: `source venv/bin/activate`.

---

### Task 1: Package scaffolding + pytest config

**Files:**
- Create: `ml/pyproject.toml`
- Create: `ml/requirements-dev.txt`
- Create: `ml/app/__init__.py`, `ml/app/rag/__init__.py`, `ml/app/eval/__init__.py`, `ml/app/ingest/__init__.py`
- Create: `ml/tests/__init__.py`, `ml/tests/unit/__init__.py`, `ml/tests/conftest.py`
- Test: `ml/tests/unit/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `app` with `app.__version__: str`; pytest markers `unit`, `integration`, `api`, `eval`.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_smoke.py
from __future__ import annotations

import app


def test_package_exposes_version():
    assert isinstance(app.__version__, str)
    assert app.__version__ != ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'` (or pytest not installed).

- [ ] **Step 3: Write minimal implementation**

```toml
# ml/pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = [
    "unit: fast tests using fakes only",
    "integration: hits real Pinecone/Cohere/Ollama; skipped without keys",
    "api: FastAPI endpoint tests",
    "eval: eval-regression threshold tests",
]
```

```text
# ml/requirements-dev.txt
pytest>=8.0.0
pydantic-settings>=2.1.0
```

```python
# ml/app/__init__.py
__version__ = "0.1.0"
```

Create the remaining `__init__.py` files (`app/rag`, `app/eval`, `app/ingest`, `tests`, `tests/unit`) as empty files, and an empty `ml/tests/conftest.py`.

Install dev deps: `cd ml && source venv/bin/activate && pip install -r requirements-dev.txt`

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_smoke.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/pyproject.toml ml/requirements-dev.txt ml/app ml/tests
git commit -m "chore(ml): scaffold app package and pytest config"
```

---

### Task 2: Settings (config.py)

**Files:**
- Create: `ml/app/config.py`
- Test: `ml/tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` (pydantic-settings `BaseSettings`) with fields `corpus_dir: str`, `eval_dir: str`, `reports_dir: str`, `embedding_model: str = "BAAI/bge-large-en-v1.5"`, `embedding_dim: int = 1024`, `pinecone_index: str = "clearrx-drug-interactions"`, `pinecone_namespace: str = "curated"`, `gen_model: str = "llama3.1"`, `judge_provider: str = "ollama"`, `judge_model: str = "llama3.1"`; plus `get_settings() -> Settings`.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_config.py
from __future__ import annotations

from app.config import Settings, get_settings


def test_defaults():
    s = Settings()
    assert s.embedding_dim == 1024
    assert s.pinecone_namespace == "curated"
    assert s.corpus_dir.endswith("corpus")


def test_env_override(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    monkeypatch.setenv("JUDGE_MODEL", "sonnet")
    s = Settings()
    assert s.embedding_dim == 384
    assert s.judge_model == "sonnet"


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/config.py
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

_DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    corpus_dir: str = os.path.normpath(os.path.join(_DATA, "corpus"))
    eval_dir: str = os.path.normpath(os.path.join(_DATA, "eval"))
    reports_dir: str = os.path.normpath(os.path.join(_DATA, os.pardir, "eval", "reports"))

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    pinecone_index: str = "clearrx-drug-interactions"
    pinecone_namespace: str = "curated"

    gen_model: str = "llama3.1"
    judge_provider: str = "ollama"
    judge_model: str = "llama3.1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_config.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/config.py ml/tests/unit/test_config.py
git commit -m "feat(ml): add typed Settings with env overrides"
```

---

### Task 3: Core models (Monograph, Chunk)

**Files:**
- Create: `ml/app/rag/models.py`
- Test: `ml/tests/unit/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Evidence(citation: str, url: str)`
  - `Monograph(id, drug_a, drug_b, drug_a_aliases: list[str], drug_b_aliases: list[str], drug_class_a, drug_class_b, severity, sections: dict[str, str], evidence: list[Evidence])` with validator: `severity in {"high","moderate","low"}`, non-empty `sections`. Method `all_drug_names() -> list[str]` (generic + brand aliases, lowercased).
  - `Chunk(text: str, source_doc_id: str, section: str, chunk_index: int, metadata: dict)`.
  - `Monograph` is the concrete "Document" type referenced in the spec.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_models.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.rag.models import Chunk, Evidence, Monograph


def _mono(**over):
    base = dict(
        id="int_warfarin_ibuprofen",
        drug_a="warfarin",
        drug_b="ibuprofen",
        drug_a_aliases=["Coumadin"],
        drug_b_aliases=["Advil", "Motrin"],
        drug_class_a="anticoagulant",
        drug_class_b="nsaid",
        severity="high",
        sections={"summary": "Increased bleeding risk."},
        evidence=[Evidence(citation="Ann Pharmacother 2004", url="https://example.org/1")],
    )
    base.update(over)
    return Monograph(**base)


def test_valid_monograph_and_drug_names():
    m = _mono()
    names = m.all_drug_names()
    assert "warfarin" in names and "coumadin" in names and "advil" in names
    assert all(n == n.lower() for n in names)


def test_rejects_bad_severity():
    with pytest.raises(ValidationError):
        _mono(severity="critical")


def test_rejects_empty_sections():
    with pytest.raises(ValidationError):
        _mono(sections={})


def test_chunk_defaults_metadata():
    c = Chunk(text="hi", source_doc_id="d1", section="summary", chunk_index=0)
    assert c.metadata == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.models'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/rag/models.py
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

_SEVERITIES = {"high", "moderate", "low"}


class Evidence(BaseModel):
    citation: str
    url: str


class Monograph(BaseModel):
    id: str
    drug_a: str
    drug_b: str
    drug_a_aliases: list[str] = Field(default_factory=list)
    drug_b_aliases: list[str] = Field(default_factory=list)
    drug_class_a: str
    drug_class_b: str
    severity: str
    sections: dict[str, str]
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("severity")
    @classmethod
    def _severity_ok(cls, v: str) -> str:
        if v not in _SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(_SEVERITIES)}")
        return v

    @field_validator("sections")
    @classmethod
    def _sections_nonempty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("sections must not be empty")
        return v

    def all_drug_names(self) -> list[str]:
        names = [self.drug_a, self.drug_b, *self.drug_a_aliases, *self.drug_b_aliases]
        seen: list[str] = []
        for n in names:
            low = n.lower().strip()
            if low and low not in seen:
                seen.append(low)
        return seen


class Chunk(BaseModel):
    text: str
    source_doc_id: str
    section: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_models.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/rag/models.py ml/tests/unit/test_models.py
git commit -m "feat(ml): add Monograph and Chunk domain models"
```

---

### Task 4: Corpus loader (corpus.py)

**Files:**
- Create: `ml/app/rag/corpus.py`
- Test: `ml/tests/unit/test_corpus.py`

**Interfaces:**
- Consumes: `Monograph` (Task 3).
- Produces: `load_corpus(corpus_dir: str) -> list[Monograph]` — reads every `*.json` file in the dir (one monograph per file), validates each, raises `CorpusError(path, reason)` on malformed/duplicate-id input, returns sorted-by-id. Also `CorpusError(Exception)`.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_corpus.py
from __future__ import annotations

import json

import pytest

from app.rag.corpus import CorpusError, load_corpus


def _write(dirpath, name, obj):
    p = dirpath / name
    p.write_text(json.dumps(obj))
    return p


def _doc(id_):
    return dict(
        id=id_, drug_a="warfarin", drug_b="ibuprofen",
        drug_a_aliases=[], drug_b_aliases=[],
        drug_class_a="anticoagulant", drug_class_b="nsaid",
        severity="high", sections={"summary": "x"}, evidence=[],
    )


def test_loads_and_sorts(tmp_path):
    _write(tmp_path, "b.json", _doc("int_b"))
    _write(tmp_path, "a.json", _doc("int_a"))
    docs = load_corpus(str(tmp_path))
    assert [d.id for d in docs] == ["int_a", "int_b"]


def test_rejects_duplicate_ids(tmp_path):
    _write(tmp_path, "one.json", _doc("int_dup"))
    _write(tmp_path, "two.json", _doc("int_dup"))
    with pytest.raises(CorpusError):
        load_corpus(str(tmp_path))


def test_rejects_malformed(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    with pytest.raises(CorpusError):
        load_corpus(str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.corpus'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/rag/corpus.py
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
            raw = json.loads(open(path, encoding="utf-8").read())
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_corpus.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/rag/corpus.py ml/tests/unit/test_corpus.py
git commit -m "feat(ml): add corpus loader with validation"
```

---

### Task 5: Chunker protocol + FixedSizeChunker

**Files:**
- Create: `ml/app/rag/chunking.py`
- Test: `ml/tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `Monograph`, `Chunk` (Task 3).
- Produces:
  - `Chunker` Protocol: attribute `name: str`, method `chunk(doc: Monograph) -> list[Chunk]`.
  - `FixedSizeChunker(chunk_size: int = 512, overlap: int = 64)` — splits each section's text into word windows of `chunk_size` words with `overlap` word overlap; every `Chunk` carries `metadata={"drugs_mentioned": doc.all_drug_names(), "drug_class": [drug_class_a, drug_class_b], "severity": severity}` and monotonically increasing `chunk_index` across the whole doc.
  - `chunk_metadata(doc: Monograph) -> dict` helper (reused by later chunkers).

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_chunking.py
from __future__ import annotations

from app.rag.chunking import FixedSizeChunker, chunk_metadata
from app.rag.models import Monograph


def _mono():
    words = " ".join(f"w{i}" for i in range(20))
    return Monograph(
        id="d1", drug_a="warfarin", drug_b="ibuprofen",
        drug_a_aliases=["coumadin"], drug_b_aliases=[],
        drug_class_a="anticoagulant", drug_class_b="nsaid",
        severity="high", sections={"summary": words, "mechanism": "short text"},
    )


def test_windows_have_overlap_and_indices():
    chunks = FixedSizeChunker(chunk_size=8, overlap=2).chunk(_mono())
    summary = [c for c in chunks if c.section == "summary"]
    # 20 words, window 8, step 6 -> windows starting at 0,6,12,18 => 4 chunks
    assert len(summary) == 4
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # overlap: last 2 words of window 0 equal first 2 words of window 1
    assert summary[0].text.split()[-2:] == summary[1].text.split()[:2]


def test_metadata_propagates():
    md = chunk_metadata(_mono())
    assert md["severity"] == "high"
    assert "coumadin" in md["drugs_mentioned"]
    assert md["drug_class"] == ["anticoagulant", "nsaid"]


def test_short_section_is_single_chunk():
    chunks = FixedSizeChunker(chunk_size=8, overlap=2).chunk(_mono())
    mech = [c for c in chunks if c.section == "mechanism"]
    assert len(mech) == 1 and mech[0].text == "short text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_chunking.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.chunking'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/rag/chunking.py
from __future__ import annotations

from typing import Protocol

from app.rag.models import Chunk, Monograph


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Monograph) -> list[Chunk]: ...


def chunk_metadata(doc: Monograph) -> dict:
    return {
        "drugs_mentioned": doc.all_drug_names(),
        "drug_class": [doc.drug_class_a, doc.drug_class_b],
        "severity": doc.severity,
    }


class FixedSizeChunker:
    name = "fixed"

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Monograph) -> list[Chunk]:
        md = chunk_metadata(doc)
        step = self.chunk_size - self.overlap
        chunks: list[Chunk] = []
        idx = 0
        for section, text in doc.sections.items():
            words = text.split()
            if not words:
                continue
            start = 0
            while start < len(words):
                window = words[start : start + self.chunk_size]
                chunks.append(
                    Chunk(
                        text=" ".join(window),
                        source_doc_id=doc.id,
                        section=section,
                        chunk_index=idx,
                        metadata=dict(md),
                    )
                )
                idx += 1
                if start + self.chunk_size >= len(words):
                    break
                start += step
        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_chunking.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/rag/chunking.py ml/tests/unit/test_chunking.py
git commit -m "feat(ml): add Chunker protocol and FixedSizeChunker"
```

---

### Task 6: Eval dataset model + loader

**Files:**
- Create: `ml/app/eval/dataset.py`
- Test: `ml/tests/unit/test_dataset.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EvalQuery(id, query, query_type, expected_doc_ids: list[str], expected_retrieval_topics: list[str], expected_answer_facts: list[str], must_not_say: list[str], severity)` with validator `query_type in {"interaction","dosage","contraindication"}`.
  - `load_queries(path: str) -> list[EvalQuery]` — reads a JSON file `{"queries": [...]}`, validates each, raises `DatasetError` on malformed input or duplicate ids.
  - `DatasetError(Exception)`.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_dataset.py
from __future__ import annotations

import json

import pytest

from app.eval.dataset import DatasetError, EvalQuery, load_queries


def _q(id_="q001", **over):
    base = dict(
        id=id_, query="Can I take ibuprofen with warfarin?",
        query_type="interaction", expected_doc_ids=["int_warfarin_ibuprofen"],
        expected_retrieval_topics=["NSAID anticoagulant"],
        expected_answer_facts=["Increased bleeding risk"],
        must_not_say=["safe to combine"], severity="high",
    )
    base.update(over)
    return base


def test_loads_queries(tmp_path):
    p = tmp_path / "queries.json"
    p.write_text(json.dumps({"queries": [_q(), _q("q002")]}))
    qs = load_queries(str(p))
    assert len(qs) == 2 and isinstance(qs[0], EvalQuery)


def test_rejects_bad_query_type(tmp_path):
    p = tmp_path / "queries.json"
    p.write_text(json.dumps({"queries": [_q(query_type="banana")]}))
    with pytest.raises(DatasetError):
        load_queries(str(p))


def test_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "queries.json"
    p.write_text(json.dumps({"queries": [_q("dup"), _q("dup")]}))
    with pytest.raises(DatasetError):
        load_queries(str(p))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.dataset'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/eval/dataset.py
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
        raw = json.loads(open(path, encoding="utf-8").read())
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_dataset.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/eval/dataset.py ml/tests/unit/test_dataset.py
git commit -m "feat(ml): add eval dataset model and loader"
```

---

### Task 7: Retrieval metrics (metrics.py)

**Files:**
- Create: `ml/app/eval/metrics.py`
- Test: `ml/tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: nothing (pure functions).
- Produces:
  - `retrieval_coverage(expected_topics: list[str], retrieved_texts: list[str]) -> float` — fraction of topics whose lowercased string appears as a substring in any retrieved text (the source doc's original metric).
  - `precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`
  - `recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float`
  - `reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float`
  - `ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float` (binary relevance)
  - `percentile(values: list[float], p: float) -> float` (linear interpolation; `p` in 0..100)

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_metrics.py
from __future__ import annotations

import math

from app.eval.metrics import (
    ndcg_at_k, percentile, precision_at_k, recall_at_k,
    reciprocal_rank, retrieval_coverage,
)


def test_retrieval_coverage():
    topics = ["NSAID anticoagulant", "monitor INR"]
    texts = ["... nsaid anticoagulant bleeding ...", "... watch closely ..."]
    assert retrieval_coverage(topics, texts) == 0.5
    assert retrieval_coverage([], texts) == 0.0


def test_precision_recall_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5   # 2 of top 4
    assert recall_at_k(retrieved, relevant, 4) == 2 / 3    # 2 of 3 relevant
    assert precision_at_k(retrieved, relevant, 2) == 0.5   # 1 of top 2


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5   # first hit at rank 2
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_at_k():
    # relevant at ranks 1 and 3 -> DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # ideal (2 relevant) IDCG = 1 + 1/log2(3) = 1 + 0.6309 = 1.6309
    got = ndcg_at_k(["a", "x", "c"], {"a", "c"}, 3)
    assert math.isclose(got, 1.5 / (1 + 1 / math.log2(3)), rel_tol=1e-9)
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_percentile():
    assert percentile([10, 20, 30, 40], 50) == 25.0
    assert percentile([42], 95) == 42.0
    assert percentile([], 50) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.metrics'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/eval/metrics.py
from __future__ import annotations

import math


def retrieval_coverage(expected_topics: list[str], retrieved_texts: list[str]) -> float:
    if not expected_topics:
        return 0.0
    blob = " \n ".join(t.lower() for t in retrieved_texts)
    hits = sum(1 for topic in expected_topics if topic.lower() in blob)
    return hits / len(expected_topics)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for r in topk if r in relevant_ids) / len(topk)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(topk & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(retrieved_ids[:k], start=1):
        if r in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    rank = (p / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(s[int(rank)])
    frac = rank - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_metrics.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/eval/metrics.py ml/tests/unit/test_metrics.py
git commit -m "feat(ml): add retrieval metrics (coverage, P@k, R@k, MRR, nDCG)"
```

---

### Task 8: LLM-as-judge (judge.py)

**Files:**
- Create: `ml/app/eval/judge.py`
- Test: `ml/tests/unit/test_judge.py`

**Interfaces:**
- Consumes: nothing (LLM injected as a callable `llm(prompt: str) -> str`).
- Produces:
  - `Judge` Protocol: `score_facts(answer: str, facts: list[str]) -> list[bool]`, `check_forbidden(answer: str, must_not_say: list[str]) -> list[bool]`.
  - `LLMJudge(llm: Callable[[str], str], max_retries: int = 1)` — builds a rubric prompt, expects a JSON array of booleans (one per fact), retries `max_retries` times on parse failure, raises `JudgeError` if still unparseable or wrong length. `check_forbidden` is a deterministic substring check (no LLM) returning one bool per phrase (`True` = phrase present = violation).
  - `JudgeError(Exception)`.
  - Prompt builders `build_fact_prompt(answer, facts) -> str` (exposed for reuse/inspection).

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_judge.py
from __future__ import annotations

import pytest

from app.eval.judge import JudgeError, LLMJudge


def test_score_facts_parses_json():
    judge = LLMJudge(llm=lambda p: "[true, false, true]")
    assert judge.score_facts("ans", ["f1", "f2", "f3"]) == [True, False, True]


def test_score_facts_retries_then_succeeds():
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        return "nonsense" if calls["n"] == 1 else "[true]"

    judge = LLMJudge(llm=flaky, max_retries=1)
    assert judge.score_facts("ans", ["f1"]) == [True]
    assert calls["n"] == 2


def test_score_facts_raises_on_wrong_length():
    judge = LLMJudge(llm=lambda p: "[true]", max_retries=0)
    with pytest.raises(JudgeError):
        judge.score_facts("ans", ["f1", "f2"])


def test_check_forbidden_is_substring():
    judge = LLMJudge(llm=lambda p: "[]")
    out = judge.check_forbidden("It is safe to combine these.", ["safe to combine", "no risk"])
    assert out == [True, False]


def test_empty_facts_returns_empty_without_calling_llm():
    def boom(prompt: str) -> str:
        raise AssertionError("should not be called")

    assert LLMJudge(llm=boom).score_facts("ans", []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.judge'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/eval/judge.py
from __future__ import annotations

import json
import re
from typing import Callable, Protocol


class JudgeError(Exception):
    pass


class Judge(Protocol):
    def score_facts(self, answer: str, facts: list[str]) -> list[bool]: ...
    def check_forbidden(self, answer: str, must_not_say: list[str]) -> list[bool]: ...


def build_fact_prompt(answer: str, facts: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    return (
        "You are grading whether an ANSWER covers each FACT.\n"
        "Return ONLY a JSON array of booleans, one per fact, in order.\n\n"
        f"ANSWER:\n{answer}\n\nFACTS:\n{numbered}\n\nJSON:"
    )


def _parse_bools(text: str) -> list[bool]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise JudgeError(f"no JSON array in judge output: {text!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(x, bool) for x in data):
        raise JudgeError(f"expected list of booleans, got {data!r}")
    return data


class LLMJudge:
    def __init__(self, llm: Callable[[str], str], max_retries: int = 1) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def score_facts(self, answer: str, facts: list[str]) -> list[bool]:
        if not facts:
            return []
        prompt = build_fact_prompt(answer, facts)
        last_err: JudgeError | None = None
        for _ in range(self.max_retries + 1):
            try:
                result = _parse_bools(self.llm(prompt))
            except JudgeError as exc:
                last_err = exc
                continue
            if len(result) != len(facts):
                last_err = JudgeError(f"expected {len(facts)} bools, got {len(result)}")
                continue
            return result
        raise last_err or JudgeError("judge failed")

    def check_forbidden(self, answer: str, must_not_say: list[str]) -> list[bool]:
        low = answer.lower()
        return [phrase.lower() in low for phrase in must_not_say]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_judge.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/eval/judge.py ml/tests/unit/test_judge.py
git commit -m "feat(ml): add pluggable LLM-as-judge with retry and forbidden-phrase check"
```

---

### Task 9: Pipeline protocol + keyword baseline

**Files:**
- Create: `ml/app/rag/pipeline.py`
- Create: `ml/app/eval/baseline.py`
- Test: `ml/tests/unit/test_baseline.py`

**Interfaces:**
- Consumes: `Monograph`, `Chunk` (Task 3), `Chunker`/`FixedSizeChunker` (Task 5).
- Produces:
  - In `pipeline.py`: `RagPipeline` Protocol — `retrieve(query: str, k: int) -> list[Chunk]`, `generate(query: str, chunks: list[Chunk]) -> str`.
  - In `baseline.py`: `KeywordBaseline(docs: list[Monograph], chunker: Chunker | None = None)` — implements `RagPipeline`. `retrieve` scores each chunk by count of query words (lowercased, tokenized on non-alphanumerics) present in the chunk text, returns top-`k` by score (ties broken by chunk_index). `generate` returns a deterministic templated answer concatenating the top chunks' summary sentences — an honest, LLM-free stand-in for the current system's behavior.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_baseline.py
from __future__ import annotations

from app.eval.baseline import KeywordBaseline
from app.rag.models import Monograph


def _docs():
    return [
        Monograph(
            id="int_warfarin_ibuprofen", drug_a="warfarin", drug_b="ibuprofen",
            drug_class_a="anticoagulant", drug_class_b="nsaid", severity="high",
            sections={"summary": "warfarin ibuprofen increased bleeding risk monitor INR"},
        ),
        Monograph(
            id="int_metformin_contrast", drug_a="metformin", drug_b="contrast", drug_class_a="antidiabetic",
            drug_class_b="contrast", severity="moderate",
            sections={"summary": "metformin contrast lactic acidosis kidney"},
        ),
    ]


def test_retrieve_ranks_relevant_doc_first():
    bl = KeywordBaseline(_docs())
    chunks = bl.retrieve("can I take ibuprofen with warfarin bleeding", k=1)
    assert len(chunks) == 1
    assert chunks[0].source_doc_id == "int_warfarin_ibuprofen"


def test_generate_is_deterministic_and_grounded():
    bl = KeywordBaseline(_docs())
    chunks = bl.retrieve("ibuprofen warfarin", k=1)
    ans = bl.generate("ibuprofen warfarin", chunks)
    assert "bleeding" in ans.lower()
    assert bl.generate("ibuprofen warfarin", chunks) == ans  # deterministic
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.baseline'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/rag/pipeline.py
from __future__ import annotations

from typing import Protocol

from app.rag.models import Chunk


class RagPipeline(Protocol):
    def retrieve(self, query: str, k: int) -> list[Chunk]: ...
    def generate(self, query: str, chunks: list[Chunk]) -> str: ...
```

```python
# ml/app/eval/baseline.py
from __future__ import annotations

import re

from app.rag.chunking import Chunker, FixedSizeChunker
from app.rag.models import Chunk, Monograph


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}


class KeywordBaseline:
    def __init__(self, docs: list[Monograph], chunker: Chunker | None = None) -> None:
        self.chunker = chunker or FixedSizeChunker(chunk_size=512, overlap=0)
        self.chunks: list[Chunk] = []
        for doc in docs:
            self.chunks.extend(self.chunker.chunk(doc))

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        q = _tokens(query)
        scored = [
            (len(q & _tokens(c.text)), c.chunk_index, c) for c in self.chunks
        ]
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [c for score, _, c in scored[:k] if score > 0]

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant interaction information found."
        body = " ".join(c.text for c in chunks)
        return f"Based on the available information: {body}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_baseline.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/rag/pipeline.py ml/app/eval/baseline.py ml/tests/unit/test_baseline.py
git commit -m "feat(ml): add RagPipeline protocol and keyword baseline"
```

---

### Task 10: Eval runner + report (runner.py)

**Files:**
- Create: `ml/app/eval/runner.py`
- Test: `ml/tests/unit/test_runner.py`

**Interfaces:**
- Consumes: `RagPipeline` (Task 9), `Judge` (Task 8), `EvalQuery` (Task 6), metrics (Task 7).
- Produces:
  - `EvalRunner(pipeline: RagPipeline, judge: Judge, k: int = 5)` with `run(queries: list[EvalQuery]) -> EvalReport`.
  - `EvalReport` (pydantic) with `aggregate: dict` (mean `retrieval_coverage`, `precision_at_k`, `recall_at_k`, `mrr`, `ndcg`, `fact_coverage`, `forbidden_violations` count, `latency_ms_p50`, `latency_ms_p95`) and `per_query: list[dict]`. Methods `to_markdown() -> str` and `write(reports_dir: str, run_id: str) -> tuple[str, str]` (writes `<run_id>.json` and `<run_id>.md`, returns their paths). `run_id` is caller-supplied for determinism.
  - Latency is measured via an injectable `clock: Callable[[], float]` (defaults to `time.perf_counter`) so tests can supply a fake clock.

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/unit/test_runner.py
from __future__ import annotations

import json

from app.eval.dataset import EvalQuery
from app.eval.runner import EvalRunner
from app.rag.models import Chunk


class FakePipeline:
    def retrieve(self, query, k):
        return [Chunk(text="increased bleeding risk nsaid anticoagulant",
                      source_doc_id="int_warfarin_ibuprofen", section="summary", chunk_index=0)]

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
    ticks = iter([0.0, 0.010, 0.0, 0.010])  # start,end per query
    runner = EvalRunner(FakePipeline(), FakeJudge(), k=5, clock=lambda: next(ticks))
    report = runner.run([_query()])
    agg = report.aggregate
    assert agg["retrieval_coverage"] == 1.0
    assert agg["precision_at_k"] == 0.2  # 1 relevant in top-5 slots
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.runner'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ml/app/eval/runner.py
from __future__ import annotations

import json
import os
import time
from typing import Callable

from pydantic import BaseModel

from app.eval.dataset import EvalQuery
from app.eval.judge import Judge
from app.eval.metrics import (
    ndcg_at_k, percentile, precision_at_k, recall_at_k,
    reciprocal_rank, retrieval_coverage,
)
from app.rag.pipeline import RagPipeline


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


class EvalReport(BaseModel):
    aggregate: dict
    per_query: list[dict]

    def to_markdown(self) -> str:
        lines = ["| metric | value |", "| --- | --- |"]
        for key, val in self.aggregate.items():
            lines.append(f"| {key} | {round(val, 4)} |")
        return "\n".join(lines) + "\n"

    def write(self, reports_dir: str, run_id: str) -> tuple[str, str]:
        os.makedirs(reports_dir, exist_ok=True)
        json_path = os.path.join(reports_dir, f"{run_id}.json")
        md_path = os.path.join(reports_dir, f"{run_id}.md")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(self.model_dump(), fh, indent=2)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(self.to_markdown())
        return json_path, md_path


class EvalRunner:
    def __init__(
        self,
        pipeline: RagPipeline,
        judge: Judge,
        k: int = 5,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.pipeline = pipeline
        self.judge = judge
        self.k = k
        self.clock = clock

    def run(self, queries: list[EvalQuery]) -> EvalReport:
        rows: list[dict] = []
        latencies: list[float] = []
        for q in queries:
            start = self.clock()
            chunks = self.pipeline.retrieve(q.query, self.k)
            answer = self.pipeline.generate(q.query, chunks)
            latency_ms = (self.clock() - start) * 1000.0
            latencies.append(latency_ms)

            retrieved_ids = [c.source_doc_id for c in chunks]
            relevant = set(q.expected_doc_ids)
            facts = self.judge.score_facts(answer, q.expected_answer_facts)
            forbidden = self.judge.check_forbidden(answer, q.must_not_say)
            rows.append(
                {
                    "id": q.id,
                    "retrieval_coverage": retrieval_coverage(
                        q.expected_retrieval_topics, [c.text for c in chunks]
                    ),
                    "precision_at_k": precision_at_k(retrieved_ids, relevant, self.k),
                    "recall_at_k": recall_at_k(retrieved_ids, relevant, self.k),
                    "mrr": reciprocal_rank(retrieved_ids, relevant),
                    "ndcg": ndcg_at_k(retrieved_ids, relevant, self.k),
                    "fact_coverage": _mean([1.0 if f else 0.0 for f in facts]),
                    "forbidden_violations": sum(1 for v in forbidden if v),
                    "latency_ms": latency_ms,
                    "answer": answer,
                }
            )

        aggregate = {
            "retrieval_coverage": _mean([r["retrieval_coverage"] for r in rows]),
            "precision_at_k": _mean([r["precision_at_k"] for r in rows]),
            "recall_at_k": _mean([r["recall_at_k"] for r in rows]),
            "mrr": _mean([r["mrr"] for r in rows]),
            "ndcg": _mean([r["ndcg"] for r in rows]),
            "fact_coverage": _mean([r["fact_coverage"] for r in rows]),
            "forbidden_violations": float(sum(r["forbidden_violations"] for r in rows)),
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
        }
        return EvalReport(aggregate=aggregate, per_query=rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/unit/test_runner.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add ml/app/eval/runner.py ml/tests/unit/test_runner.py
git commit -m "feat(ml): add eval runner with json+markdown report output"
```

---

### Task 11: Seed corpus + eval queries + baseline run

**Files:**
- Create: `ml/data/corpus/int_warfarin_ibuprofen.json` (+ ≥7 more monographs; see list below)
- Create: `ml/data/eval/queries.json` (≥12 queries)
- Create: `ml/data/eval/human_labels.json` (≥6 hand-labeled fact judgments)
- Create: `ml/scripts/run_baseline.py`
- Test: `ml/tests/eval/__init__.py`, `ml/tests/eval/test_seed_data.py`

**Interfaces:**
- Consumes: `load_corpus` (Task 4), `load_queries` (Task 6), `KeywordBaseline` (Task 9), `EvalRunner` (Task 10), `LLMJudge` (Task 8), `get_settings` (Task 2).
- Produces: validated seed data on disk; `ml/scripts/run_baseline.py` runnable as `python -m scripts.run_baseline` writing `baseline.json`/`baseline.md` to `ml/eval/reports/`.

Author the initial corpus covering these interactions (each an original-prose monograph with all five sections + ≥1 evidence citation): warfarin+ibuprofen, warfarin+aspirin, aspirin+ibuprofen, atorvastatin+clarithromycin, metformin+iodinated-contrast, lisinopril+spironolactone, sildenafil+nitroglycerin, ssri+maoi (fluoxetine+phenelzine). These map onto common, well-documented interactions and give the eval real ranking targets. (Full corpus growth to ~60 interactions continues after Phase 0; the seed makes the harness runnable.)

- [ ] **Step 1: Write the failing test**

```python
# ml/tests/eval/test_seed_data.py
from __future__ import annotations

from app.eval.dataset import load_queries
from app.eval.judge import LLMJudge
from app.eval.baseline import KeywordBaseline
from app.eval.runner import EvalRunner
from app.rag.corpus import load_corpus


def test_seed_corpus_and_queries_load():
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    doc_ids = {d.id for d in docs}
    assert len(docs) >= 8
    assert len(queries) >= 12
    # every expected_doc_id in the eval set must exist in the corpus
    for q in queries:
        for did in q.expected_doc_ids:
            assert did in doc_ids, f"{q.id} references unknown doc {did}"


def test_baseline_runs_end_to_end_offline():
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    judge = LLMJudge(llm=lambda p: "[" + ", ".join("true" for _ in range(p.count("\n" + "1. ") or 1)) + "]")
    # simpler deterministic judge: mark everything covered
    judge = LLMJudge(llm=lambda p: "[]")  # replaced below per-call is overkill; use fake
    runner = EvalRunner(KeywordBaseline(docs), _AllTrueJudge(), k=5, clock=lambda: 0.0)
    report = runner.run(queries)
    assert 0.0 <= report.aggregate["recall_at_k"] <= 1.0
    # the keyword baseline should retrieve the right doc for at least half the queries
    assert report.aggregate["recall_at_k"] > 0.4


class _AllTrueJudge:
    def score_facts(self, answer, facts):
        return [True for _ in facts]

    def check_forbidden(self, answer, must_not_say):
        return [phrase.lower() in answer.lower() for phrase in must_not_say]
```

Note: keep only the `_AllTrueJudge` version in the file (delete the two throwaway `judge =` lines shown above — they illustrate why a fake judge is used instead of a lambda). Final test body uses `_AllTrueJudge()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/eval/test_seed_data.py -v`
Expected: FAIL — `CorpusError`/`FileNotFoundError` (no seed data yet).

- [ ] **Step 3: Write minimal implementation**

Author the 8 monograph JSON files under `ml/data/corpus/`. Template (fill each with real, concise clinical prose — original wording, not copied from a proprietary DB):

```json
{
  "id": "int_warfarin_ibuprofen",
  "drug_a": "warfarin",
  "drug_b": "ibuprofen",
  "drug_a_aliases": ["coumadin", "jantoven"],
  "drug_b_aliases": ["advil", "motrin", "nurofen"],
  "drug_class_a": "anticoagulant",
  "drug_class_b": "nsaid",
  "severity": "high",
  "sections": {
    "summary": "Combining warfarin with ibuprofen substantially increases the risk of serious bleeding, including gastrointestinal hemorrhage.",
    "mechanism": "Ibuprofen inhibits platelet cyclooxygenase, reducing platelet aggregation, and can irritate the gastric mucosa; warfarin inhibits vitamin K dependent clotting factors. The effects on hemostasis are additive.",
    "clinical_effects": "Increased bruising, GI bleeding, hematuria, and prolonged bleeding time. INR may rise modestly but bleeding risk is elevated even at a stable INR.",
    "management": "Avoid the combination when possible. Prefer acetaminophen for analgesia. If an NSAID is unavoidable, use the lowest dose for the shortest duration with gastroprotection.",
    "monitoring": "Monitor INR more frequently after starting or stopping ibuprofen, and counsel patients to report signs of bleeding promptly."
  },
  "evidence": [
    {"citation": "Battistella M, et al. Arch Intern Med. 2005", "url": "https://pubmed.ncbi.nlm.nih.gov/15824297/"}
  ]
}
```

Create `ml/data/eval/queries.json` with ≥12 queries (mix of interaction/dosage/contraindication `query_type`), each referencing a real corpus `id` in `expected_doc_ids`, with `expected_retrieval_topics`, `expected_answer_facts`, and `must_not_say` (e.g. `["safe to combine", "no interaction"]` for high-severity pairs). Example entry:

```json
{
  "queries": [
    {
      "id": "q001",
      "query": "Can I take ibuprofen with my warfarin prescription?",
      "query_type": "interaction",
      "expected_doc_ids": ["int_warfarin_ibuprofen"],
      "expected_retrieval_topics": ["nsaid anticoagulant", "bleeding risk"],
      "expected_answer_facts": [
        "Increased bleeding risk",
        "NSAIDs reduce platelet function",
        "Monitor INR if combined"
      ],
      "must_not_say": ["safe to combine", "no interaction"],
      "severity": "high"
    }
  ]
}
```

Create `ml/data/eval/human_labels.json` with ≥6 entries pairing a query id + a specific answer text + the human's per-fact booleans, for judge calibration in a later phase:

```json
{
  "labels": [
    {
      "query_id": "q001",
      "answer": "Ibuprofen and warfarin together raise bleeding risk; monitor INR.",
      "human_fact_labels": [true, false, true]
    }
  ]
}
```

Create the runnable baseline script:

```python
# ml/scripts/run_baseline.py
from __future__ import annotations

from app.config import get_settings
from app.eval.baseline import KeywordBaseline
from app.eval.dataset import load_queries
from app.eval.judge import LLMJudge
from app.eval.runner import EvalRunner
from app.rag.corpus import load_corpus


def _offline_judge(prompt: str) -> str:
    # Phase 0 has no LLM; grade every fact False so fact_coverage is a
    # conservative floor. Replaced by a real judge in Phase 1+.
    n = prompt.count(". ", prompt.find("FACTS:")) if "FACTS:" in prompt else 0
    return "[" + ", ".join("false" for _ in range(max(n, 0))) + "]"


def main() -> None:
    settings = get_settings()
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    runner = EvalRunner(KeywordBaseline(docs), LLMJudge(_offline_judge, max_retries=2), k=5)
    report = runner.run(queries)
    json_path, md_path = report.write("eval/reports", run_id="baseline")
    print(report.to_markdown())
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
```

Create empty `ml/tests/eval/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ml && source venv/bin/activate && python -m pytest tests/eval/test_seed_data.py -v`
Expected: PASS (2 passed).

Then run the whole suite and the baseline script:

Run: `cd ml && source venv/bin/activate && python -m pytest -v && python -m scripts.run_baseline`
Expected: all tests PASS; a markdown metrics table prints; `ml/eval/reports/baseline.json` and `baseline.md` are written.

- [ ] **Step 5: Commit**

```bash
git add ml/data ml/scripts ml/tests/eval ml/eval/reports/baseline.json ml/eval/reports/baseline.md
git commit -m "feat(ml): add seed corpus, eval queries, and baseline runner"
```

---

## Self-Review

**Spec coverage:**
- §2 corpus data model → Tasks 3, 4, 11. §3 stack decisions → deferred impls behind interfaces (Tasks 5, 8, 9); config names in Task 2. §4 package structure → Tasks 1–10. §5 data models → Tasks 3 (Monograph/Chunk), 6 (EvalQuery). §7 eval harness (baseline, separate retrieval/generation metrics, pluggable judge, calibration data) → Tasks 7, 8, 9, 10, 11. §9 unit-test strategy → every task. Integration/API/streaming/Pinecone/Cohere/chunking-experiments → deferred to Phase 1–4 plans (out of Phase 0 scope, stated in the plan header).
- Judge calibration *reporting* (§7) uses `human_labels.json` seeded in Task 11; the agreement computation lands in the Phase 1 plan alongside the first real judge — noted here so it isn't lost.

**Placeholder scan:** No TBD/TODO left in code steps; the one templated artifact (corpus prose) is a data-authoring task with a complete worked example, not code. Task 11's test contains an intentional teaching aside that Step 3's note instructs the implementer to delete — flagged explicitly so it isn't shipped.

**Type consistency:** `Monograph`, `Chunk`, `EvalQuery`, `EvalReport`, `RagPipeline`, `Judge`, `Chunker` names and signatures are consistent across producing/consuming tasks. `retrieve(query, k)`/`generate(query, chunks)` match between Task 9 (definition), Task 10 (runner use), and Task 11 (script). Metric function names match between Task 7 and Task 10.

## Execution Handoff

Phase 0 delivers a working, fully-tested eval harness and a runnable baseline report — with zero external dependencies — on top of which Phases 1–5 (Pinecone, chunking experiments, hybrid+rerank, streaming, debrief) each get their own plan.
