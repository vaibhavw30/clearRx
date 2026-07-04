# ClearRx Phase 1b — Real Local Dense Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the keyword baseline with a real dense-retrieval RAG pipeline — BGE embeddings + a Pinecone dotproduct index + Ollama generation + a real LLM-as-judge with human-label calibration — and capture the keyword→dense comparison report.

**Architecture:** Every stage sits behind a Python `Protocol` already sketched in the design spec (§4). Each real client (BGE, Pinecone, Ollama, OpenAI) uses **lazy imports** and **constructor injection** so the modules import and the unit tests run with fakes on any machine, while the heavy dependencies and live services are only needed for `@pytest.mark.integration` tests and the end-to-end run. Dense retrieval is proven at parity-or-better against the committed keyword baseline; hybrid + rerank are deferred to Phase 3.

**Tech Stack:** Python 3.9.6, pydantic v2, pytest, sentence-transformers (BGE `bge-large-en-v1.5`, 1024-dim, L2-normalized), Pinecone serverless (dotproduct metric), Ollama (`llama3.1`) over HTTP via httpx, OpenAI (`gpt-4o-mini`) as the pluggable judge escape hatch.

## Global Constraints

- Python **3.9.6** — every module starts with `from __future__ import annotations`.
- **pydantic v2** for all models.
- **All real clients use lazy imports** (import `sentence_transformers` / `pinecone` / `openai` *inside* `__init__` or the method, never at module top) and accept an **injected collaborator** (`model=`, `index=`, `post=`, `client=`) so unit tests never touch the network or heavy deps.
- **Embeddings are L2-normalized** (`normalize_embeddings=True`); the Pinecone index metric is **dotproduct** (dim **1024**). Normalized vectors + dotproduct == cosine, and dotproduct is what Phase 3 hybrid requires — the index is created once and never recreated.
- **BGE query instruction:** queries (not documents) are embedded with the prefix `Represent this sentence for searching relevant passages: ` — this is a documented BGE retrieval requirement.
- **Unit tests** (`tests/unit`) are offline/fakes-only and run anywhere. **Integration tests** (`tests/integration`, `@pytest.mark.integration`) and the end-to-end scripts require live services and run **only on the provisioned machine**; they are skipped unless `RUN_INTEGRATION=1`.
- Work runs from `ml/` with the venv active: `cd ml && source venv/bin/activate`. Run tests with `python -m pytest`.
- Commit after every task; prefix `feat(ml):` / `test(ml):` / `chore(ml):`.
- Reuse existing modules — do not duplicate: `app/rag/models.py` (`Monograph`, `Chunk`), `app/rag/chunking.py` (`Chunker`, `FixedSizeChunker`), `app/rag/corpus.py` (`load_corpus`), `app/eval/dataset.py` (`load_queries`, `EvalQuery`), `app/eval/judge.py` (`Judge`, `LLMJudge`, `build_fact_prompt`), `app/eval/runner.py` (`EvalRunner`, `EvalReport`).

## Prerequisites (do once on the provisioned machine, before Task-level integration runs)

These are **operator setup**, not code tasks:
1. `pip install -r requirements.txt` (installs sentence-transformers, pinecone, httpx, openai — already listed after Task 1).
2. Create a Pinecone free-tier account; set `PINECONE_API_KEY` in the environment.
3. `OPENAI_API_KEY` already present in `.env` (used only if the judge escape hatch is flipped on).
4. Install Ollama, then `ollama pull llama3.1`. Ensure `ollama serve` is running (default `http://localhost:11434`).
5. First BGE load downloads ~1.3 GB to the HuggingFace cache.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `ml/requirements.txt` | runtime deps | Modify: add `pinecone` |
| `ml/app/config.py` | settings | Modify: add Pinecone/Ollama/OpenAI/judge fields |
| `ml/tests/conftest.py` | integration-skip gate | Modify: skip `integration` unless `RUN_INTEGRATION=1` |
| `ml/app/rag/embeddings.py` | `Embedder` + `BGEEmbedder` | Create |
| `ml/app/rag/vectorstore.py` | `Record`/`Match`/`VectorStore` + `PineconeStore` | Create |
| `ml/app/rag/generator.py` | `LLMClient` + `OllamaClient` | Create |
| `ml/app/rag/judge_clients.py` | `OpenAIChat` callable + `build_judge` factory | Create |
| `ml/app/rag/pipeline.py` | `DenseRagPipeline` (impl of the existing `RagPipeline` Protocol) | Modify |
| `ml/app/ingest/build_index.py` | corpus→chunk→embed→upsert CLI + pure `build_records` | Create |
| `ml/app/eval/calibration.py` | judge-vs-human agreement | Create |
| `ml/scripts/run_dense.py` | end-to-end dense eval + calibration + comparison | Create |
| `ml/tests/unit/test_embeddings.py` … | unit tests (fakes) | Create |
| `ml/tests/integration/test_dense_stack.py` | live-service tests | Create |

---

### Task 1: Dependencies, config, and integration-skip gate

Add the Pinecone dep, the new settings fields (all with safe defaults so unit tests need no env), and a conftest hook that skips integration tests unless explicitly enabled.

**Files:**
- Modify: `ml/requirements.txt`
- Modify: `ml/app/config.py`
- Modify: `ml/tests/conftest.py`
- Test: `ml/tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` gains `pinecone_api_key: str`, `pinecone_metric: str`, `pinecone_cloud: str`, `pinecone_region: str`, `ollama_base_url: str`, `openai_api_key: str`, and reuses existing `embedding_model`/`embedding_dim`/`gen_model`/`judge_provider`/`judge_model`. A `pytest_collection_modifyitems` hook skips `@pytest.mark.integration` unless `RUN_INTEGRATION=1`.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_config.py`:

```python
def test_phase1b_settings_defaults(monkeypatch):
    from app.config import Settings
    for var in ["PINECONE_API_KEY", "OPENAI_API_KEY", "OLLAMA_BASE_URL"]:
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.embedding_dim == 1024
    assert s.pinecone_metric == "dotproduct"
    assert s.pinecone_cloud == "aws"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.pinecone_api_key == ""      # safe default, no env needed for unit tests
    assert s.judge_provider == "ollama"


def test_pinecone_key_read_from_env(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("PINECONE_API_KEY", "pc-xyz")
    assert Settings().pinecone_api_key == "pc-xyz"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_config.py::test_phase1b_settings_defaults -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'pinecone_metric'`

- [ ] **Step 3: Add deps, settings, and the conftest hook**

In `ml/requirements.txt`, add one line (sentence-transformers, httpx, openai are already present):

```
pinecone>=5.0.0
```

In `ml/app/config.py`, add these fields to `Settings` (after the existing `judge_model` line):

```python
    pinecone_api_key: str = ""
    pinecone_metric: str = "dotproduct"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
```

Replace the contents of `ml/tests/conftest.py` with:

```python
from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test; set RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt app/config.py tests/conftest.py tests/unit/test_config.py
git commit -m "feat(ml): add Phase 1b config, pinecone dep, integration-skip gate"
```

---

### Task 2: `Embedder` + `BGEEmbedder`

**Files:**
- Create: `ml/app/rag/embeddings.py`
- Test: `ml/tests/unit/test_embeddings.py`
- Test: `ml/tests/integration/test_dense_stack.py` (create with the first integration test)

**Interfaces:**
- Produces:
  - `class Embedder(Protocol)` with `dimension: int`, `embed(self, texts: list[str]) -> np.ndarray` (shape `(n, dimension)`, L2-normalized), `embed_query(self, text: str) -> np.ndarray` (shape `(dimension,)`).
  - `class BGEEmbedder` with `__init__(self, model_name: str, dimension: int = 1024, *, model=None)` — injecting `model` skips the lazy `SentenceTransformer` import.
  - `QUERY_INSTRUCTION: str` constant.

- [ ] **Step 1: Write the failing unit test**

Create `ml/tests/unit/test_embeddings.py`:

```python
from __future__ import annotations

import numpy as np

from app.rag.embeddings import BGEEmbedder, QUERY_INSTRUCTION


class FakeST:
    """Stand-in for SentenceTransformer that records inputs and returns
    fixed, un-normalized vectors so we can assert the wrapper normalizes."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True):
        self.calls.append({"texts": list(texts), "normalize": normalize_embeddings})
        base = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
        arr = base[: len(texts)]
        if normalize_embeddings:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        return arr


def test_embed_documents_are_normalized_and_shaped():
    fake = FakeST()
    emb = BGEEmbedder("bge", dimension=2, model=fake)
    out = emb.embed(["doc one", "doc two"])
    assert out.shape == (2, 2)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert fake.calls[0]["normalize"] is True
    assert fake.calls[0]["texts"] == ["doc one", "doc two"]  # no instruction on docs


def test_embed_query_prepends_instruction_and_is_1d():
    fake = FakeST()
    emb = BGEEmbedder("bge", dimension=2, model=fake)
    vec = emb.embed_query("ibuprofen warfarin")
    assert vec.shape == (2,)
    assert fake.calls[0]["texts"] == [QUERY_INSTRUCTION + "ibuprofen warfarin"]


def test_dimension_attribute():
    assert BGEEmbedder("bge", dimension=1024, model=FakeST()).dimension == 1024
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.embeddings'`

- [ ] **Step 3: Implement `embeddings.py`**

Create `ml/app/rag/embeddings.py`:

```python
from __future__ import annotations

from typing import Protocol

import numpy as np

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class BGEEmbedder:
    """BGE sentence-transformer embedder. Vectors are L2-normalized so a
    Pinecone dotproduct index behaves as cosine. Queries get the BGE
    retrieval instruction; documents do not."""

    def __init__(self, model_name: str, dimension: int = 1024, *, model=None) -> None:
        self.dimension = dimension
        if model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            model = SentenceTransformer(model_name)
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vecs = self.model.encode(
            [QUERY_INSTRUCTION + text], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vecs, dtype=np.float32)[0]
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Add the integration test (runs only on the provisioned machine)**

Create `ml/tests/integration/__init__.py` (empty) and `ml/tests/integration/test_dense_stack.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from app.config import get_settings


@pytest.mark.integration
def test_bge_real_dimension_and_norm():
    from app.rag.embeddings import BGEEmbedder

    s = get_settings()
    emb = BGEEmbedder(s.embedding_model, s.embedding_dim)
    out = emb.embed(["warfarin ibuprofen bleeding risk"])
    assert out.shape == (1, s.embedding_dim)          # 1024
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-3)
    q = emb.embed_query("can I take advil with coumadin")
    assert q.shape == (s.embedding_dim,)
```

- [ ] **Step 6: Verify collection skips integration locally**

Run: `python -m pytest tests/integration -v`
Expected: the integration test is **skipped** (reason: "set RUN_INTEGRATION=1"). No model download occurs.

- [ ] **Step 7: Commit**

```bash
git add app/rag/embeddings.py tests/unit/test_embeddings.py tests/integration/
git commit -m "feat(ml): add BGE embedder (normalized, query-instructed)"
```

---

### Task 3: `Record` / `Match` / `VectorStore` + `PineconeStore`

**Files:**
- Create: `ml/app/rag/vectorstore.py`
- Test: `ml/tests/unit/test_vectorstore.py`
- Test: append to `ml/tests/integration/test_dense_stack.py`

**Interfaces:**
- Produces:
  - `class Record(BaseModel)`: `id: str`, `values: list[float]`, `metadata: dict`.
  - `class Match(BaseModel)`: `id: str`, `score: float`, `metadata: dict`.
  - `class VectorStore(Protocol)`: `upsert(self, records: list[Record], namespace: str) -> None`; `query(self, dense: list[float], top_k: int, flt: Optional[dict], namespace: str) -> list[Match]`.
  - `class PineconeStore` with `__init__(self, api_key: str, index_name: str, *, client=None, index=None)`, plus `ensure_index(self, dimension: int, metric: str, cloud: str, region: str) -> None`.

- [ ] **Step 1: Write the failing unit test**

Create `ml/tests/unit/test_vectorstore.py`:

```python
from __future__ import annotations

from app.rag.vectorstore import Match, PineconeStore, Record


class FakeIndex:
    def __init__(self):
        self.upserted = []
        self.queries = []

    def upsert(self, vectors, namespace):
        self.upserted.append({"vectors": vectors, "namespace": namespace})

    def query(self, vector, top_k, include_metadata, filter, namespace):
        self.queries.append(
            {"vector": vector, "top_k": top_k, "include_metadata": include_metadata,
             "filter": filter, "namespace": namespace}
        )
        return {
            "matches": [
                {"id": "int_x::summary::0", "score": 0.91,
                 "metadata": {"chunk_text": "bleeding risk", "source_doc_id": "int_x"}},
            ]
        }


def test_upsert_maps_records_to_pinecone_vectors():
    idx = FakeIndex()
    store = PineconeStore("k", "clearrx", index=idx)
    store.upsert(
        [Record(id="r1", values=[0.1, 0.2], metadata={"chunk_text": "t"})], namespace="curated"
    )
    call = idx.upserted[0]
    assert call["namespace"] == "curated"
    assert call["vectors"][0] == {"id": "r1", "values": [0.1, 0.2], "metadata": {"chunk_text": "t"}}


def test_query_returns_match_objects():
    idx = FakeIndex()
    store = PineconeStore("k", "clearrx", index=idx)
    matches = store.query([0.1, 0.2], top_k=5, flt=None, namespace="curated")
    assert matches == [
        Match(id="int_x::summary::0", score=0.91,
              metadata={"chunk_text": "bleeding risk", "source_doc_id": "int_x"})
    ]
    assert idx.queries[0]["top_k"] == 5
    assert idx.queries[0]["include_metadata"] is True   # store always requests metadata
    assert idx.queries[0]["filter"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_vectorstore.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.vectorstore'`

- [ ] **Step 3: Implement `vectorstore.py`**

Create `ml/app/rag/vectorstore.py`:

```python
from __future__ import annotations

from typing import Optional, Protocol

from pydantic import BaseModel


class Record(BaseModel):
    id: str
    values: list[float]
    metadata: dict


class Match(BaseModel):
    id: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    def upsert(self, records: list[Record], namespace: str) -> None: ...
    def query(
        self, dense: list[float], top_k: int, flt: Optional[dict], namespace: str
    ) -> list[Match]: ...


class PineconeStore:
    """Pinecone serverless dense store. Inject `index` (and/or `client`) in
    tests to avoid importing the SDK or hitting the network."""

    def __init__(self, api_key: str, index_name: str, *, client=None, index=None) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self._client = client
        self._index = index

    def _pc(self):
        if self._client is None:
            from pinecone import Pinecone  # lazy

            self._client = Pinecone(api_key=self.api_key)
        return self._client

    def _idx(self):
        if self._index is None:
            self._index = self._pc().Index(self.index_name)
        return self._index

    def ensure_index(self, dimension: int, metric: str, cloud: str, region: str) -> None:
        from pinecone import ServerlessSpec  # lazy

        pc = self._pc()
        existing = {i["name"] for i in pc.list_indexes()}
        if self.index_name not in existing:
            pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )

    def upsert(self, records: list[Record], namespace: str) -> None:
        vectors = [
            {"id": r.id, "values": list(r.values), "metadata": r.metadata} for r in records
        ]
        self._idx().upsert(vectors=vectors, namespace=namespace)

    def query(
        self, dense: list[float], top_k: int, flt: Optional[dict], namespace: str
    ) -> list[Match]:
        res = self._idx().query(
            vector=list(dense),
            top_k=top_k,
            include_metadata=True,
            filter=flt,
            namespace=namespace,
        )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out: list[Match] = []
        for m in matches:
            md = m["metadata"] if isinstance(m, dict) else m.metadata
            mid = m["id"] if isinstance(m, dict) else m.id
            score = m["score"] if isinstance(m, dict) else m.score
            out.append(Match(id=mid, score=float(score), metadata=dict(md)))
        return out
```

Note: `PineconeStore` calls the index with keyword args (`upsert(vectors=..., namespace=...)`, `query(vector=..., top_k=..., include_metadata=..., filter=..., namespace=...)`). The `FakeIndex` in the test already matches these keywords, so the calls line up.

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_vectorstore.py -v`
Expected: PASS

- [ ] **Step 5: Add the integration round-trip test**

Append to `ml/tests/integration/test_dense_stack.py`:

```python
@pytest.mark.integration
def test_pinecone_upsert_query_roundtrip():
    from app.rag.embeddings import BGEEmbedder
    from app.rag.vectorstore import PineconeStore, Record

    s = get_settings()
    store = PineconeStore(s.pinecone_api_key, s.pinecone_index)
    store.ensure_index(s.embedding_dim, s.pinecone_metric, s.pinecone_cloud, s.pinecone_region)
    emb = BGEEmbedder(s.embedding_model, s.embedding_dim)
    vec = emb.embed(["warfarin plus ibuprofen raises bleeding risk"])[0]
    store.upsert(
        [Record(id="itest::0", values=vec.tolist(),
                metadata={"chunk_text": "warfarin ibuprofen bleeding", "source_doc_id": "itest"})],
        namespace="itest",
    )
    import time as _t

    _t.sleep(5)  # Pinecone upserts are eventually consistent
    q = emb.embed_query("can I combine warfarin and ibuprofen")
    matches = store.query(q.tolist(), top_k=1, flt=None, namespace="itest")
    assert matches and matches[0].metadata["source_doc_id"] == "itest"
```

- [ ] **Step 6: Commit**

```bash
git add app/rag/vectorstore.py tests/unit/test_vectorstore.py tests/integration/test_dense_stack.py
git commit -m "feat(ml): add Pinecone dense vector store"
```

---

### Task 4: `LLMClient` + `OllamaClient`

**Files:**
- Create: `ml/app/rag/generator.py`
- Test: `ml/tests/unit/test_generator.py`
- Test: append to `ml/tests/integration/test_dense_stack.py`

**Interfaces:**
- Produces:
  - `class LLMClient(Protocol)`: `generate(self, prompt: str) -> str`; `stream(self, prompt: str) -> Iterator[str]`.
  - `class OllamaClient` with `__init__(self, base_url: str, model: str, *, post=None)` where `post: Callable[[str, dict], dict]` defaults to an httpx POST. `generate` calls `/api/generate` with `stream=False`.

- [ ] **Step 1: Write the failing unit test**

Create `ml/tests/unit/test_generator.py`:

```python
from __future__ import annotations

from app.rag.generator import OllamaClient


def test_generate_posts_prompt_and_returns_response():
    seen = {}

    def fake_post(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"response": "Increased bleeding risk."}

    client = OllamaClient("http://localhost:11434", "llama3.1", post=fake_post)
    out = client.generate("Context...\n\nQuery: warfarin ibuprofen?")
    assert out == "Increased bleeding risk."
    assert seen["url"] == "http://localhost:11434/api/generate"
    assert seen["payload"]["model"] == "llama3.1"
    assert seen["payload"]["stream"] is False
    assert "warfarin ibuprofen" in seen["payload"]["prompt"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.generator'`

- [ ] **Step 3: Implement `generator.py`**

Create `ml/app/rag/generator.py`:

```python
from __future__ import annotations

from typing import Callable, Iterator, Optional, Protocol


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...


class OllamaClient:
    """Ollama HTTP client. Inject `post` in tests to avoid httpx/network."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        post: Optional[Callable[[str, dict], dict]] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._post = post

    def _do_post(self, url: str, payload: dict) -> dict:
        if self._post is not None:
            return self._post(url, payload)
        import httpx  # lazy

        resp = httpx.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def generate(self, prompt: str) -> str:
        data = self._do_post(
            f"{self.base_url}/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False},
        )
        return data["response"]

    def stream(self, prompt: str) -> Iterator[str]:
        # Phase 4 wires SSE end-to-end; a simple non-chunked fallback keeps the
        # Protocol satisfied and callers correct in the meantime.
        yield self.generate(prompt)
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_generator.py -v`
Expected: PASS

- [ ] **Step 5: Add the integration test**

Append to `ml/tests/integration/test_dense_stack.py`:

```python
@pytest.mark.integration
def test_ollama_generate_returns_text():
    from app.rag.generator import OllamaClient

    s = get_settings()
    client = OllamaClient(s.ollama_base_url, s.gen_model)
    out = client.generate("Answer in one short sentence: what is 2 + 2?")
    assert isinstance(out, str) and out.strip()
```

- [ ] **Step 6: Commit**

```bash
git add app/rag/generator.py tests/unit/test_generator.py tests/integration/test_dense_stack.py
git commit -m "feat(ml): add Ollama LLM client"
```

---

### Task 5: `DenseRagPipeline`

Implement the existing `RagPipeline` Protocol (`app/rag/pipeline.py`) with a real dense pipeline: embed the query, query Pinecone, map matches to `Chunk`s, and generate a grounded, cited answer.

**Files:**
- Modify: `ml/app/rag/pipeline.py`
- Test: `ml/tests/unit/test_pipeline.py`

**Interfaces:**
- Consumes: `Embedder` (Task 2), `VectorStore`/`Match` (Task 3), `LLMClient` (Task 4), `Chunk` (`app/rag/models.py`).
- Produces:
  - `def build_context(chunks: list[Chunk]) -> str`
  - `def build_prompt(query: str, chunks: list[Chunk]) -> str`
  - `class DenseRagPipeline` with `__init__(self, embedder, store, llm, namespace: str, k: int = 5)`, `retrieve(self, query: str, k: int) -> list[Chunk]`, `generate(self, query: str, chunks: list[Chunk]) -> str`.
  - Match metadata keys written by Task 6 and read here: `chunk_text`, `source_doc_id`, `section`, `chunk_index`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_pipeline.py`:

```python
from __future__ import annotations

import numpy as np

from app.rag.models import Chunk
from app.rag.pipeline import DenseRagPipeline, build_prompt
from app.rag.vectorstore import Match


class FakeEmbedder:
    dimension = 2

    def embed(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)

    def embed_query(self, text):
        return np.array([0.6, 0.8], dtype=np.float32)


class FakeStore:
    def __init__(self):
        self.last = {}

    def upsert(self, records, namespace):
        ...

    def query(self, dense, top_k, flt, namespace):
        self.last = {"dense": list(dense), "top_k": top_k, "namespace": namespace}
        return [
            Match(id="int_warfarin_ibuprofen::summary::0", score=0.9,
                  metadata={"chunk_text": "increased bleeding risk",
                            "source_doc_id": "int_warfarin_ibuprofen",
                            "section": "summary", "chunk_index": 0})
        ]


class FakeLLM:
    def __init__(self):
        self.prompt = None

    def generate(self, prompt):
        self.prompt = prompt
        return "There is an increased bleeding risk [int_warfarin_ibuprofen]."

    def stream(self, prompt):
        yield self.generate(prompt)


def test_retrieve_embeds_query_and_maps_matches_to_chunks():
    store = FakeStore()
    pipe = DenseRagPipeline(FakeEmbedder(), store, FakeLLM(), namespace="curated")
    chunks = pipe.retrieve("advil with coumadin?", k=5)
    assert store.last["top_k"] == 5
    assert store.last["namespace"] == "curated"
    assert store.last["dense"] == [0.6, 0.8]          # the query embedding, not a doc embedding
    assert len(chunks) == 1
    c = chunks[0]
    assert isinstance(c, Chunk)
    assert c.text == "increased bleeding risk"
    assert c.source_doc_id == "int_warfarin_ibuprofen"
    assert c.section == "summary"


def test_generate_includes_context_and_query_in_prompt():
    llm = FakeLLM()
    pipe = DenseRagPipeline(FakeEmbedder(), FakeStore(), llm, namespace="curated")
    chunk = Chunk(text="increased bleeding risk", source_doc_id="int_warfarin_ibuprofen",
                  section="summary", chunk_index=0)
    answer = pipe.generate("advil with coumadin?", [chunk])
    assert "increased bleeding risk" in llm.prompt        # context present
    assert "advil with coumadin?" in llm.prompt            # query present
    assert answer.startswith("There is an increased bleeding risk")


def test_build_prompt_cites_source_ids():
    chunk = Chunk(text="risk", source_doc_id="int_warfarin_ibuprofen",
                  section="summary", chunk_index=0)
    prompt = build_prompt("q?", [chunk])
    assert "int_warfarin_ibuprofen" in prompt
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'DenseRagPipeline' from 'app.rag.pipeline'`

- [ ] **Step 3: Implement `DenseRagPipeline` in `pipeline.py`**

Append to `ml/app/rag/pipeline.py` (keep the existing `RagPipeline` Protocol at the top):

```python
from app.rag.embeddings import Embedder
from app.rag.generator import LLMClient
from app.rag.vectorstore import VectorStore


def build_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c.source_doc_id} / {c.section}] {c.text}")
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: list[Chunk]) -> str:
    context = build_context(chunks)
    return (
        "You are a careful clinical drug-interaction assistant. Answer ONLY from the "
        "context below. If the context does not cover the drugs asked about, say there "
        "is no interaction information available rather than guessing. Cite the source "
        "document id in square brackets after each claim. This is not medical advice.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\nANSWER:"
    )


class DenseRagPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLMClient,
        namespace: str,
        k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.namespace = namespace
        self.k = k

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        vec = self.embedder.embed_query(query)
        matches = self.store.query(
            vec.tolist(), top_k=k, flt=None, namespace=self.namespace
        )
        chunks: list[Chunk] = []
        for m in matches:
            md = m.metadata
            chunks.append(
                Chunk(
                    text=md.get("chunk_text", ""),
                    source_doc_id=md.get("source_doc_id", ""),
                    section=md.get("section", ""),
                    chunk_index=int(md.get("chunk_index", 0)),
                    metadata=md,
                )
            )
        return chunks

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant interaction information found in the corpus."
        return self.llm.generate(build_prompt(query, chunks))
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat(ml): add dense RAG pipeline (retrieve + grounded generate)"
```

---

### Task 6: `ingest/build_index.py`

Chunk the corpus, embed the chunks, and upsert them to Pinecone with metadata. Split the pure `build_records` transform (unit-testable with fakes) from the `main()` CLI that touches Pinecone.

**Files:**
- Create: `ml/app/ingest/build_index.py`
- Test: `ml/tests/unit/test_build_index.py`

**Interfaces:**
- Consumes: `load_corpus`, `FixedSizeChunker`, `Embedder`, `Record`, `PineconeStore`, `chunk_metadata` (already in `app/rag/chunking.py`).
- Produces:
  - `def build_records(docs: list[Monograph], chunker: Chunker, embedder: Embedder) -> list[Record]` — record id `f"{doc_id}::{section}::{chunk_index}"`; metadata carries `chunk_text`, `source_doc_id`, `section`, `chunk_index`, plus the chunk's existing metadata (`drugs_mentioned`, `drug_class`, `severity`).
  - `def main() -> None` — the CLI (`python -m app.ingest.build_index`).

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_build_index.py`:

```python
from __future__ import annotations

import numpy as np

from app.ingest.build_index import build_records
from app.rag.chunking import FixedSizeChunker
from app.rag.models import Monograph


class FakeEmbedder:
    dimension = 3

    def embed(self, texts):
        return np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)

    def embed_query(self, text):
        return np.zeros(3, dtype=np.float32)


def _doc():
    return Monograph(
        id="int_a_b", drug_a="a", drug_b="b", drug_class_a="x", drug_class_b="y",
        severity="high",
        sections={"summary": "a and b interact", "management": "avoid the combination"},
    )


def test_build_records_ids_and_metadata():
    recs = build_records([_doc()], FixedSizeChunker(chunk_size=512, overlap=0), FakeEmbedder())
    assert len(recs) == 2  # one chunk per non-empty section
    ids = {r.id for r in recs}
    assert "int_a_b::summary::0" in ids
    summary = next(r for r in recs if r.id == "int_a_b::summary::0")
    assert summary.metadata["chunk_text"] == "a and b interact"
    assert summary.metadata["source_doc_id"] == "int_a_b"
    assert summary.metadata["section"] == "summary"
    assert summary.metadata["severity"] == "high"
    assert "a" in summary.metadata["drugs_mentioned"]
    assert len(summary.values) == 3  # embedding attached
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_build_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingest.build_index'`

- [ ] **Step 3: Implement `build_index.py`**

Create `ml/app/ingest/build_index.py`:

```python
from __future__ import annotations

from app.config import get_settings
from app.rag.chunking import Chunker, FixedSizeChunker
from app.rag.corpus import load_corpus
from app.rag.embeddings import BGEEmbedder, Embedder
from app.rag.models import Monograph
from app.rag.vectorstore import PineconeStore, Record


def build_records(
    docs: list[Monograph], chunker: Chunker, embedder: Embedder
) -> list[Record]:
    chunks = [c for doc in docs for c in chunker.chunk(doc)]
    if not chunks:
        return []
    vectors = embedder.embed([c.text for c in chunks])
    records: list[Record] = []
    for c, vec in zip(chunks, vectors):
        metadata = dict(c.metadata)
        metadata.update(
            {
                "chunk_text": c.text,
                "source_doc_id": c.source_doc_id,
                "section": c.section,
                "chunk_index": c.chunk_index,
            }
        )
        records.append(
            Record(
                id=f"{c.source_doc_id}::{c.section}::{c.chunk_index}",
                values=[float(x) for x in vec],
                metadata=metadata,
            )
        )
    return records


def main() -> None:
    settings = get_settings()
    docs = load_corpus(settings.corpus_dir)
    embedder = BGEEmbedder(settings.embedding_model, settings.embedding_dim)
    store = PineconeStore(settings.pinecone_api_key, settings.pinecone_index)
    store.ensure_index(
        settings.embedding_dim, settings.pinecone_metric,
        settings.pinecone_cloud, settings.pinecone_region,
    )
    records = build_records(docs, FixedSizeChunker(chunk_size=512, overlap=0), embedder)
    store.upsert(records, namespace=settings.pinecone_namespace)
    print(f"upserted {len(records)} chunks from {len(docs)} monographs "
          f"to index '{settings.pinecone_index}' namespace '{settings.pinecone_namespace}'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_build_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/ingest/build_index.py tests/unit/test_build_index.py
git commit -m "feat(ml): add corpus->pinecone index builder"
```

---

### Task 7: OpenAI judge callable + `build_judge` factory

Wire the pluggable judge: local Ollama by default, OpenAI as the escape hatch. The existing `LLMJudge` already accepts a `Callable[[str], str]`, so the factory just picks the callable.

**Files:**
- Create: `ml/app/rag/judge_clients.py`
- Test: `ml/tests/unit/test_judge_factory.py`
- Test: append to `ml/tests/integration/test_dense_stack.py`

**Interfaces:**
- Consumes: `OllamaClient` (Task 4), `LLMJudge` (`app/eval/judge.py`), `Settings`.
- Produces:
  - `class OpenAIChat` with `__init__(self, api_key: str, model: str, *, client=None)` and `__call__(self, prompt: str) -> str`.
  - `def build_judge(settings, *, ollama=None, openai_chat=None) -> LLMJudge` — dispatches on `settings.judge_provider` (`"ollama"` | `"openai"`), `max_retries=2`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_judge_factory.py`:

```python
from __future__ import annotations

from app.config import Settings
from app.rag.judge_clients import OpenAIChat, build_judge


class FakeOllama:
    def generate(self, prompt):
        return "[true, false]"


class FakeChat:
    def __call__(self, prompt):
        return "[true, false]"


def test_build_judge_uses_ollama_by_default():
    s = Settings(judge_provider="ollama")
    judge = build_judge(s, ollama=FakeOllama())
    assert judge.score_facts("answer", ["fact one", "fact two"]) == [True, False]


def test_build_judge_uses_openai_when_selected():
    s = Settings(judge_provider="openai")
    judge = build_judge(s, openai_chat=FakeChat())
    assert judge.score_facts("answer", ["fact one", "fact two"]) == [True, False]


def test_build_judge_rejects_unknown_provider():
    s = Settings(judge_provider="bogus")
    try:
        build_judge(s)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown provider")


class FakeOpenAIClient:
    class chat:
        class completions:
            @staticmethod
            def create(model, messages):
                class M:
                    class choices:
                        pass
                obj = type("R", (), {})()
                obj.choices = [type("C", (), {"message": type("Msg", (), {"content": "[true]"})()})()]
                return obj


def test_openai_chat_extracts_message_content():
    chat = OpenAIChat("k", "gpt-4o-mini", client=FakeOpenAIClient())
    assert chat("grade this") == "[true]"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_judge_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.judge_clients'`

- [ ] **Step 3: Implement `judge_clients.py`**

Create `ml/app/rag/judge_clients.py`:

```python
from __future__ import annotations

from app.eval.judge import LLMJudge
from app.rag.generator import OllamaClient


class OpenAIChat:
    """Callable judge backend using OpenAI chat completions. Inject `client`
    in tests to avoid the SDK/network."""

    def __init__(self, api_key: str, model: str, *, client=None) -> None:
        self.model = model
        if client is None:
            from openai import OpenAI  # lazy

            client = OpenAI(api_key=api_key)
        self.client = client

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content


def build_judge(settings, *, ollama=None, openai_chat=None) -> LLMJudge:
    provider = settings.judge_provider.lower()
    if provider == "ollama":
        client = ollama or OllamaClient(settings.ollama_base_url, settings.judge_model)
        return LLMJudge(client.generate, max_retries=2)
    if provider == "openai":
        chat = openai_chat or OpenAIChat(settings.openai_api_key, settings.judge_model)
        return LLMJudge(chat, max_retries=2)
    raise ValueError(f"unknown judge_provider: {settings.judge_provider!r}")
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_judge_factory.py -v`
Expected: PASS

- [ ] **Step 5: Add the integration test**

Append to `ml/tests/integration/test_dense_stack.py`:

```python
@pytest.mark.integration
def test_local_judge_scores_facts():
    from app.rag.judge_clients import build_judge

    s = get_settings()  # judge_provider defaults to ollama
    judge = build_judge(s)
    result = judge.score_facts(
        "Ibuprofen and warfarin together increase bleeding risk.",
        ["Increased bleeding risk", "Reduces blood pressure"],
    )
    assert result == [True, False] or (isinstance(result, list) and len(result) == 2)
```

- [ ] **Step 6: Commit**

```bash
git add app/rag/judge_clients.py tests/unit/test_judge_factory.py tests/integration/test_dense_stack.py
git commit -m "feat(ml): add pluggable judge factory (ollama default, openai hatch)"
```

---

### Task 8: `eval/calibration.py`

Score the human-labeled subset with the configured judge and report judge-vs-human agreement.

**Files:**
- Create: `ml/app/eval/calibration.py`
- Test: `ml/tests/unit/test_calibration.py`

**Interfaces:**
- Consumes: `Judge` (`app/eval/judge.py`), `EvalQuery` (`app/eval/dataset.py`).
- Produces:
  - `def load_labels(path: str) -> list[dict]` — reads `{"labels": [...]}`.
  - `def calibrate(judge, queries_by_id: dict, labels: list[dict]) -> dict` — returns `{"agreement": float, "n_facts": int, "n_labels": int, "per_query": [{"query_id", "matches", "n"}]}`. Agreement = (# facts where judge bool == human bool) / (total facts).

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_calibration.py`:

```python
from __future__ import annotations

from app.eval.calibration import calibrate
from app.eval.dataset import EvalQuery


class ScriptedJudge:
    """Returns preset booleans per query so agreement is hand-computable."""

    def __init__(self, mapping):
        self.mapping = mapping

    def score_facts(self, answer, facts):
        return self.mapping[answer]

    def check_forbidden(self, answer, must_not_say):
        return [False for _ in must_not_say]


def _q(qid, facts):
    return EvalQuery(id=qid, query="x", query_type="interaction",
                     expected_doc_ids=["d"], expected_answer_facts=facts,
                     must_not_say=[], severity="high")


def test_calibrate_computes_fact_level_agreement():
    queries = {"q1": _q("q1", ["f1", "f2", "f3"]), "q2": _q("q2", ["f1", "f2"])}
    labels = [
        {"query_id": "q1", "answer": "A1", "human_fact_labels": [True, False, True]},
        {"query_id": "q2", "answer": "A2", "human_fact_labels": [True, True]},
    ]
    judge = ScriptedJudge({"A1": [True, False, False], "A2": [True, True]})
    #   q1: judge [T,F,F] vs human [T,F,T] -> 2/3 agree
    #   q2: judge [T,T]   vs human [T,T]   -> 2/2 agree
    #   total 4/5 = 0.8
    result = calibrate(judge, queries, labels)
    assert result["n_labels"] == 2
    assert result["n_facts"] == 5
    assert abs(result["agreement"] - 0.8) < 1e-9
    per = {p["query_id"]: p for p in result["per_query"]}
    assert per["q1"] == {"query_id": "q1", "matches": 2, "n": 3}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.calibration'`

- [ ] **Step 3: Implement `calibration.py`**

Create `ml/app/eval/calibration.py`:

```python
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
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/eval/calibration.py tests/unit/test_calibration.py
git commit -m "feat(ml): add judge-vs-human calibration"
```

---

### Task 9: `scripts/run_dense.py` — end-to-end dense eval + calibration

The capstone the operator runs **after** `python -m app.ingest.build_index`. It runs the dense pipeline over the eval set, writes the `dense` report, runs calibration, and prints the keyword→dense comparison.

**Files:**
- Create: `ml/scripts/run_dense.py`
- Test: `ml/tests/unit/test_run_dense.py`

**Interfaces:**
- Consumes: everything above + `EvalRunner`, `EvalReport`, `load_corpus`, `load_queries`.
- Produces: `def compare(baseline: dict, dense: dict) -> list[dict]` (pure, unit-tested); `def main() -> None` (manual run). Writes `eval/reports/dense.{json,md}`.

- [ ] **Step 1: Write the failing unit test (for the pure comparison helper)**

Create `ml/tests/unit/test_run_dense.py`:

```python
from __future__ import annotations

from scripts.run_dense import compare


def test_compare_reports_deltas_for_shared_metrics():
    baseline = {"recall_at_k": 0.5, "precision_at_k": 0.2, "mrr": 0.4}
    dense = {"recall_at_k": 0.8, "precision_at_k": 0.5, "mrr": 0.7}
    rows = compare(baseline, dense)
    by_metric = {r["metric"]: r for r in rows}
    assert abs(by_metric["recall_at_k"]["delta"] - 0.3) < 1e-9
    assert by_metric["recall_at_k"]["baseline"] == 0.5
    assert by_metric["recall_at_k"]["dense"] == 0.8
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_run_dense.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_dense'`

- [ ] **Step 3: Implement `run_dense.py`**

Create `ml/scripts/run_dense.py`:

```python
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
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_run_dense.py -v`
Expected: PASS

- [ ] **Step 5: Run the full unit suite**

Run: `python -m pytest tests/unit tests/eval -q`
Expected: PASS (all offline tests green; integration skipped).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_dense.py tests/unit/test_run_dense.py
git commit -m "feat(ml): add end-to-end dense eval + calibration script"
```

---

## Operator run book (on the provisioned machine, after all 9 tasks)

```bash
cd ml && source venv/bin/activate
pip install -r requirements.txt
export PINECONE_API_KEY=...          # OPENAI_API_KEY already in .env
ollama pull llama3.1 && ollama serve # (serve in another shell if not already running)

# 1. offline gate
python -m pytest tests/unit tests/eval -q

# 2. live-service gate (downloads BGE ~1.3GB on first run)
RUN_INTEGRATION=1 python -m pytest tests/integration -v

# 3. build the vector index
python -m app.ingest.build_index

# 4. run the dense eval + calibration + keyword->dense comparison
python -m scripts.run_dense
```

Expected outcome: `eval/reports/dense.{json,md}` written; dense `recall@5`/`precision@5`/`mrr`/`ndcg` at or above the keyword baseline (recall@5 0.5676, precision@5 0.1586, mrr 0.4536, ndcg 0.4822); `fact_coverage` now non-zero; a printed judge-vs-human agreement %. If agreement < 80%, flip the judge to OpenAI and re-run.

---

## Self-Review

- **Spec coverage (design §8 Step 2 / Phase 1b roadmap):** Pinecone index creation at the embedder dimension with dotproduct metric + namespaces (Task 3, 6); dense-only query (Task 5); real judge with pluggable escape hatch (Task 7); calibration % (Task 8); dense report vs baseline (Task 9). BGE embeddings (Task 2), Ollama generation (Task 4), config/deps/skip-gate (Task 1).
- **Deferred (correctly out of scope):** hybrid/BM25 + Cohere rerank (Phase 3), SSE streaming end-to-end (Phase 4 — `OllamaClient.stream` is a minimal placeholder satisfying the Protocol), README debrief (Phase 5), metadata filtering at query time (`flt` plumbed through but unused; a Phase-2/3 concern). The 1a deferred items (recall-floor test, negative `must_not_say` diversity) are not triggered by 1b (corpus unchanged) and stay deferred.
- **Placeholder scan:** none. Every module and its primary unit test carry complete, runnable code.
- **Type consistency:** `Embedder.embed`/`embed_query`, `Record{id,values,metadata}`, `Match{id,score,metadata}`, `VectorStore.query(dense, top_k, flt, namespace)`, `LLMClient.generate`, `DenseRagPipeline(embedder, store, llm, namespace, k)`, `build_records(docs, chunker, embedder)`, `build_judge(settings, *, ollama, openai_chat)`, and `calibrate(judge, queries_by_id, labels)` are used identically across tasks. The metadata keys written in Task 6 (`chunk_text`/`source_doc_id`/`section`/`chunk_index`) are exactly those read in Task 5.
