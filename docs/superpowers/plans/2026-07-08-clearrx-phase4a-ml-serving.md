# ClearRx Phase 4a — RAG Serving Layer (ML) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up `app/main.py` as the single ML FastAPI service — `/query` + `/query/stream` (SSE) over the RAG pipeline with grounded citations, plus RAG-backed `/interactions/check-enhanced` and corpus-backed `/drugs` that preserve the dashboard's response contracts — and delete the legacy FAISS `ml/main.py`.

**Architecture:** A thin FastAPI app whose routes call injected dependencies (`get_retriever`, `get_llm`) so API tests run offline with fakes via `app.dependency_overrides`. Business logic stays in `rag/` (`build_prompt`) and pure helpers in `app/answer.py`. Streaming uses `StreamingResponse` with `text/event-stream`; the assembled token stream provably equals the non-streamed answer. This is Phase 4a of the streaming feature; Express passthrough (4b) and the frontend UI (4c) follow in their own plans.

**Tech Stack:** Python 3.9.6, pydantic v2, FastAPI + Starlette `TestClient`, pytest.

## Global Constraints

- Python **3.9.6** — every module starts with `from __future__ import annotations`.
- **pydantic v2** for all models.
- **Injectable + offline-testable:** routes depend on `get_retriever`/`get_llm`; API tests override them with fakes (`app.dependency_overrides`) and never construct BGE/Pinecone/Ollama. Heavy deps stay lazy (they already are in `embeddings`/`vectorstore`/`generator`).
- **Preserve response contracts:** `InteractionResponse` fields are exactly `severity: str, description: str, recommendation: str, sources: list[str], confidence: float, method: str`; `EnhancedInteractionRequest` is `drugA: str, drugB: str, patientContext: dict`. `/drugs` returns `{"drugs": [...], "count": int}`. These match what the dashboard/Express already consume.
- **Citations are grounded in retrieval**, never invented by the LLM.
- Reuse existing modules — do NOT duplicate: `rag/pipeline.py` (`build_prompt`, `build_context`, `DenseRagPipeline`), `rag/generator.py` (`LLMClient`, `OllamaClient`), `rag/embeddings.py` (`BGEEmbedder`), `rag/vectorstore.py` (`PineconeStore`), `rag/corpus.py` (`load_corpus`), `rag/models.py` (`Chunk`, `Monograph`), `config.py` (`get_settings`).
- Work runs from `ml/` with the venv active: `cd ml && source venv/bin/activate`. Run tests with `python -m pytest`. New API tests live in `tests/api/` and use the `api` pytest marker (already declared in `pyproject.toml`). Commit after every task.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `ml/app/schemas.py` | pydantic request/response models | Create |
| `ml/app/answer.py` | pure answer/citation/SSE helpers | Create |
| `ml/app/deps.py` | FastAPI dependencies (retriever, llm) | Create |
| `ml/app/main.py` | thin FastAPI app + routes | Create |
| `ml/main.py` | legacy FAISS app | Delete (Task 6) |
| `ml/tests/api/` | FastAPI endpoint tests | Create |
| `ml/tests/unit/test_answer.py` | answer-helper unit tests | Create |

---

### Task 1: Schemas

**Files:**
- Create: `ml/app/schemas.py`
- Test: `ml/tests/unit/test_schemas.py`

**Interfaces:**
- Produces: `QueryRequest{query: str, top_k: int = 5}`, `Citation{source_doc_id: str, section: str | None = None, url: str | None = None}`, `QueryResponse{answer: str, citations: list[Citation]}`, `HealthResponse{status: str, services: dict}`, `EnhancedInteractionRequest{drugA: str, drugB: str, patientContext: dict = {}}`, `InteractionResponse{severity: str, description: str, recommendation: str, sources: list[str] = [], confidence: float = 0.0, method: str}`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_schemas.py`:

```python
from __future__ import annotations

from app.schemas import (
    Citation, EnhancedInteractionRequest, InteractionResponse,
    QueryRequest, QueryResponse,
)


def test_query_request_defaults_top_k():
    assert QueryRequest(query="warfarin ibuprofen").top_k == 5


def test_query_response_nests_citations():
    r = QueryResponse(answer="a", citations=[Citation(source_doc_id="int_x")])
    assert r.citations[0].source_doc_id == "int_x"
    assert r.citations[0].url is None


def test_interaction_response_contract_fields():
    r = InteractionResponse(severity="high", description="d", recommendation="r", method="rag")
    assert r.sources == [] and r.confidence == 0.0 and r.method == "rag"


def test_enhanced_interaction_request_patient_context_optional():
    req = EnhancedInteractionRequest(drugA="warfarin", drugB="ibuprofen")
    assert req.patientContext == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.schemas'`

- [ ] **Step 3: Implement**

Create `ml/app/schemas.py`:

```python
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class Citation(BaseModel):
    source_doc_id: str
    section: Optional[str] = None
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    services: dict = Field(default_factory=dict)


class EnhancedInteractionRequest(BaseModel):
    drugA: str
    drugB: str
    patientContext: dict = Field(default_factory=dict)


class InteractionResponse(BaseModel):
    severity: str
    description: str
    recommendation: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    method: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/unit/test_schemas.py
git commit -m "feat(ml): add API request/response schemas"
```

---

### Task 2: Answer helpers

**Files:**
- Create: `ml/app/answer.py`
- Test: `ml/tests/unit/test_answer.py`

**Interfaces:**
- Consumes: `Chunk` (`app.rag.models`), `Citation`/`InteractionResponse` (`app.schemas`).
- Produces:
  - `citations_from_chunks(chunks) -> list[Citation]` — distinct by `source_doc_id`, first-seen order; `section` from the chunk, `url` from `chunk.metadata.get("source_url")`.
  - `sse_frame(data: str, event: str | None = None) -> str` — `"event: {event}\n"` prefix when given, then `"data: {data}\n\n"`.
  - `build_interaction_response(chunks, answer: str) -> InteractionResponse` — see rules below.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_answer.py`:

```python
from __future__ import annotations

from app.answer import build_interaction_response, citations_from_chunks, sse_frame
from app.rag.models import Chunk


def _chunk(doc, section, url=None, severity="high", text="t"):
    md = {"source_url": url, "severity": severity} if url else {"severity": severity}
    return Chunk(text=text, source_doc_id=doc, section=section, chunk_index=0, metadata=md)


def test_citations_distinct_first_seen_with_url():
    chunks = [
        _chunk("int_a", "summary", url="http://x"),
        _chunk("int_a", "management"),   # same doc -> collapsed
        _chunk("int_b", "summary", url="http://y"),
    ]
    cites = citations_from_chunks(chunks)
    assert [c.source_doc_id for c in cites] == ["int_a", "int_b"]
    assert cites[0].url == "http://x" and cites[0].section == "summary"


def test_sse_frame_with_and_without_event():
    assert sse_frame("hello") == "data: hello\n\n"
    assert sse_frame("[1]", event="citations") == "event: citations\ndata: [1]\n\n"


def test_build_interaction_response_maps_from_chunks():
    chunks = [
        _chunk("int_warfarin_ibuprofen", "summary", url="http://s", severity="high"),
        _chunk("int_warfarin_ibuprofen", "management", text="Avoid the combination."),
    ]
    r = build_interaction_response(chunks, answer="Increased bleeding risk.")
    assert r.severity == "high"
    assert r.description == "Increased bleeding risk."
    assert r.recommendation == "Avoid the combination."   # from the management-section chunk
    assert r.sources == ["http://s"]                       # url preferred over id
    assert r.method == "rag" and r.confidence > 0.0


def test_build_interaction_response_no_chunks_is_safe():
    r = build_interaction_response([], answer="")
    assert r.severity == "unknown" and r.sources == [] and r.confidence == 0.0
    assert "Consult" in r.recommendation and r.method == "rag"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_answer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.answer'`

- [ ] **Step 3: Implement**

Create `ml/app/answer.py`:

```python
from __future__ import annotations

from typing import Optional

from app.rag.models import Chunk
from app.schemas import Citation, InteractionResponse

_FALLBACK_RECOMMENDATION = (
    "Consult a pharmacist or physician before combining these medications."
)


def citations_from_chunks(chunks: list[Chunk]) -> list[Citation]:
    cites: list[Citation] = []
    seen: set = set()
    for c in chunks:
        if c.source_doc_id in seen:
            continue
        seen.add(c.source_doc_id)
        cites.append(
            Citation(
                source_doc_id=c.source_doc_id,
                section=c.section or None,
                url=c.metadata.get("source_url"),
            )
        )
    return cites


def sse_frame(data: str, event: Optional[str] = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def build_interaction_response(chunks: list[Chunk], answer: str) -> InteractionResponse:
    if not chunks:
        return InteractionResponse(
            severity="unknown",
            description="No interaction information available in the corpus.",
            recommendation=_FALLBACK_RECOMMENDATION,
            sources=[],
            confidence=0.0,
            method="rag",
        )
    severity = chunks[0].metadata.get("severity", "unknown")
    management = next((c.text for c in chunks if c.section == "management"), None)
    cites = citations_from_chunks(chunks)
    sources = [c.url or c.source_doc_id for c in cites]
    return InteractionResponse(
        severity=severity,
        description=answer,
        recommendation=management or _FALLBACK_RECOMMENDATION,
        sources=sources,
        confidence=0.85,
        method="rag",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_answer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/answer.py tests/unit/test_answer.py
git commit -m "feat(ml): add citation + SSE + interaction-mapping answer helpers"
```

---

### Task 3: FastAPI app — `/health` + `/query`

**Files:**
- Create: `ml/app/deps.py`
- Create: `ml/app/main.py`
- Test: `ml/tests/api/__init__.py` (empty), `ml/tests/api/test_query.py`

**Interfaces:**
- Consumes: `build_prompt` (`app.rag.pipeline`), `citations_from_chunks` (`app.answer`), schemas, `get_settings`, `DenseRagPipeline`, `BGEEmbedder`, `PineconeStore`, `OllamaClient`.
- Produces:
  - `app/deps.py`: `get_retriever()` → `DenseRagPipeline(BGEEmbedder(...), PineconeStore(...), llm=None, namespace=settings.pinecone_namespace)` (only `.retrieve` is used; `llm=None` is fine — retrieve ignores it); `get_llm()` → `OllamaClient(settings.ollama_base_url, settings.gen_model)`.
  - `app/main.py`: `app = FastAPI(...)`; `GET /health` → `HealthResponse`; `POST /query` (body `QueryRequest`) → `QueryResponse`. Routes depend on `get_retriever`/`get_llm` via `Depends`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/api/__init__.py` (empty) and `ml/tests/api/test_query.py`:

```python
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.deps import get_llm, get_retriever
from app.main import app
from app.rag.models import Chunk

pytestmark = pytest.mark.api


class FakeRetriever:
    def retrieve(self, query, k):
        return [Chunk(text="increased bleeding risk", source_doc_id="int_warfarin_ibuprofen",
                      section="summary", chunk_index=0,
                      metadata={"source_url": "http://x", "severity": "high"})]


class FakeLLM:
    def generate(self, prompt):
        return "There is an increased bleeding risk. [int_warfarin_ibuprofen]"
    def stream(self, prompt):
        yield "There is an increased "
        yield "bleeding risk. [int_warfarin_ibuprofen]"


class EmptyRetriever:
    def retrieve(self, query, k):
        return []


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_ok():
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_query_returns_answer_and_grounded_citations(client):
    r = client.post("/query", json={"query": "warfarin ibuprofen"})
    assert r.status_code == 200
    body = r.json()
    assert "bleeding risk" in body["answer"]
    assert body["citations"][0]["source_doc_id"] == "int_warfarin_ibuprofen"
    assert body["citations"][0]["url"] == "http://x"


def test_query_no_chunks_returns_no_info_answer():
    app.dependency_overrides[get_retriever] = lambda: EmptyRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    try:
        r = TestClient(app).post("/query", json={"query": "aspirin water"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200
    assert r.json()["citations"] == []
    assert "no interaction information" in r.json()["answer"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_query.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement**

Create `ml/app/deps.py`:

```python
from __future__ import annotations

from app.config import get_settings
from app.rag.embeddings import BGEEmbedder
from app.rag.generator import OllamaClient
from app.rag.pipeline import DenseRagPipeline
from app.rag.vectorstore import PineconeStore


def get_retriever():
    """Dense retriever for query-time retrieval. `.retrieve` is all the routes
    use; `llm=None` is safe because DenseRagPipeline.retrieve ignores it.
    Overridden with a fake in tests."""
    s = get_settings()
    embedder = BGEEmbedder(s.embedding_model, s.embedding_dim)
    store = PineconeStore(s.pinecone_api_key, s.pinecone_index)
    return DenseRagPipeline(embedder, store, llm=None, namespace=s.pinecone_namespace)


def get_llm():
    """LLM client for generation/streaming. Overridden with a fake in tests."""
    s = get_settings()
    return OllamaClient(s.ollama_base_url, s.gen_model)
```

Create `ml/app/main.py`:

```python
from __future__ import annotations

from fastapi import Depends, FastAPI

from app.answer import citations_from_chunks
from app.deps import get_llm, get_retriever
from app.rag.pipeline import build_prompt
from app.schemas import HealthResponse, QueryRequest, QueryResponse

app = FastAPI(title="ClearRx RAG service")

_NO_INFO = "No interaction information available in the corpus."


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", services={"rag": True})


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, retriever=Depends(get_retriever), llm=Depends(get_llm)) -> QueryResponse:
    chunks = retriever.retrieve(req.query, req.top_k)
    if not chunks:
        return QueryResponse(answer=_NO_INFO, citations=[])
    answer = llm.generate(build_prompt(req.query, chunks))
    return QueryResponse(answer=answer, citations=citations_from_chunks(chunks))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_query.py -v`
Expected: PASS (health + both query tests)

- [ ] **Step 5: Commit**

```bash
git add app/deps.py app/main.py tests/api/__init__.py tests/api/test_query.py
git commit -m "feat(ml): add FastAPI app with /health and /query routes"
```

---

### Task 4: `/query/stream` (SSE)

**Files:**
- Modify: `ml/app/main.py`
- Test: `ml/tests/api/test_stream.py`

**Interfaces:**
- Consumes: `sse_frame` (`app.answer`), `StreamingResponse` (`fastapi.responses`), `json`.
- Produces: `POST /query/stream` → `StreamingResponse(media_type="text/event-stream")` yielding one `data:` frame per `llm.stream(prompt)` token, then an `event: citations` frame (JSON array of citation dicts), then `data: [DONE]`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/api/test_stream.py`:

```python
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.deps import get_llm, get_retriever
from app.main import app
from app.rag.models import Chunk

pytestmark = pytest.mark.api


class FakeRetriever:
    def retrieve(self, query, k):
        return [Chunk(text="bleeding", source_doc_id="int_warfarin_ibuprofen",
                      section="summary", chunk_index=0,
                      metadata={"source_url": "http://x", "severity": "high"})]


class FakeLLM:
    _CHUNKS = ["There is an increased ", "bleeding risk. [int_warfarin_ibuprofen]"]
    def generate(self, prompt):
        return "".join(self._CHUNKS)
    def stream(self, prompt):
        for c in self._CHUNKS:
            yield c


def _tokens_from_sse(text: str) -> str:
    """Reassemble the answer from data: frames, ignoring the citations event and [DONE]."""
    out = []
    for block in text.split("\n\n"):
        if not block or block.startswith("event:"):
            continue
        line = block[len("data: "):] if block.startswith("data: ") else ""
        if line and line != "[DONE]":
            out.append(line)
    return "".join(out)


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_stream_is_event_stream_and_matches_nonstream_answer(client):
    stream = client.post("/query/stream", json={"query": "warfarin ibuprofen"})
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    body = stream.text
    # assembled tokens equal the non-streamed answer
    nonstream = client.post("/query", json={"query": "warfarin ibuprofen"}).json()["answer"]
    assert _tokens_from_sse(body) == nonstream
    # citations event present and grounded; terminal [DONE]
    assert "event: citations" in body
    cite_block = [b for b in body.split("\n\n") if b.startswith("event: citations")][0]
    cites = json.loads(cite_block.split("data: ", 1)[1])
    assert cites[0]["source_doc_id"] == "int_warfarin_ibuprofen"
    assert body.rstrip().endswith("data: [DONE]")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_stream.py -v`
Expected: FAIL — 404 (route not defined) / assertion error.

- [ ] **Step 3: Implement**

In `ml/app/main.py`, add imports and the route:

```python
import json

from fastapi.responses import StreamingResponse

from app.answer import sse_frame
```

```python
@app.post("/query/stream")
def query_stream(req: QueryRequest, retriever=Depends(get_retriever), llm=Depends(get_llm)):
    chunks = retriever.retrieve(req.query, req.top_k)
    citations = citations_from_chunks(chunks)
    prompt = build_prompt(req.query, chunks) if chunks else ""

    def event_stream():
        if not chunks:
            yield sse_frame(_NO_INFO)
        else:
            for token in llm.stream(prompt):
                yield sse_frame(token)
        yield sse_frame(json.dumps([c.model_dump() for c in citations]), event="citations")
        yield sse_frame("[DONE]")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_stream.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/api/test_stream.py
git commit -m "feat(ml): add SSE /query/stream endpoint (stream == non-stream answer)"
```

---

### Task 5: RAG-backed `/interactions/check-enhanced` + corpus `/drugs`

**Files:**
- Modify: `ml/app/main.py`
- Test: `ml/tests/api/test_interactions.py`

**Interfaces:**
- Consumes: `build_interaction_response` (`app.answer`), `build_prompt`, `load_corpus` (`app.rag.corpus`), `get_settings`, schemas.
- Produces:
  - `POST /interactions/check-enhanced` (body `EnhancedInteractionRequest`) → `InteractionResponse`: builds query `f"{drugA} with {drugB} interaction"`, `retriever.retrieve` → `llm.generate(build_prompt(...))` → `build_interaction_response(chunks, answer)`.
  - `GET /drugs` → `{"drugs": [{"name": n} for n in sorted distinct corpus drug names], "count": N}` via `load_corpus(settings.corpus_dir)` + `Monograph.all_drug_names()`.
  - `GET /drugs/{drug_name}` → the first monograph mentioning that (lower-cased) name, or `404`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/api/test_interactions.py`:

```python
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.deps import get_llm, get_retriever
from app.main import app
from app.rag.models import Chunk

pytestmark = pytest.mark.api


class FakeRetriever:
    def retrieve(self, query, k):
        return [
            Chunk(text="Increased bleeding risk.", source_doc_id="int_warfarin_ibuprofen",
                  section="summary", chunk_index=0,
                  metadata={"source_url": "http://s", "severity": "high"}),
            Chunk(text="Avoid the combination; prefer acetaminophen.",
                  source_doc_id="int_warfarin_ibuprofen", section="management", chunk_index=1,
                  metadata={"severity": "high"}),
        ]


class FakeLLM:
    def generate(self, prompt):
        return "Combining them raises bleeding risk."
    def stream(self, prompt):
        yield "Combining them raises bleeding risk."


@pytest.fixture
def client():
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_interactions_check_enhanced_rag_backed(client):
    r = client.post("/interactions/check-enhanced",
                    json={"drugA": "warfarin", "drugB": "ibuprofen", "patientContext": {"age": 70}})
    assert r.status_code == 200
    body = r.json()
    assert body["severity"] == "high"
    assert body["description"] == "Combining them raises bleeding risk."
    assert body["recommendation"].startswith("Avoid the combination")
    assert body["method"] == "rag"
    assert body["sources"] == ["http://s"]


def test_drugs_served_from_corpus():
    # no override: reads the real committed corpus (offline)
    with TestClient(app) as c:
        r = c.get("/drugs")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] > 0 and len(body["drugs"]) == body["count"]
    names = {d["name"] for d in body["drugs"]}
    assert "warfarin" in names


def test_drug_detail_404_for_unknown():
    with TestClient(app) as c:
        r = c.get("/drugs/notadrug")
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_interactions.py -v`
Expected: FAIL — 404 (routes not defined).

- [ ] **Step 3: Implement**

In `ml/app/main.py`, add imports and routes:

```python
from fastapi import HTTPException

from app.answer import build_interaction_response
from app.config import get_settings
from app.rag.corpus import load_corpus
from app.schemas import EnhancedInteractionRequest, InteractionResponse
```

```python
@app.post("/interactions/check-enhanced", response_model=InteractionResponse)
def check_interaction_enhanced(
    req: EnhancedInteractionRequest, retriever=Depends(get_retriever), llm=Depends(get_llm)
) -> InteractionResponse:
    query = f"{req.drugA} with {req.drugB} interaction"
    chunks = retriever.retrieve(query, 5)
    answer = llm.generate(build_prompt(query, chunks)) if chunks else ""
    return build_interaction_response(chunks, answer)


@app.get("/drugs")
def list_drugs() -> dict:
    docs = load_corpus(get_settings().corpus_dir)
    names = sorted({n for d in docs for n in d.all_drug_names()})
    return {"drugs": [{"name": n} for n in names], "count": len(names)}


@app.get("/drugs/{drug_name}")
def get_drug(drug_name: str) -> dict:
    key = drug_name.lower().strip()
    docs = load_corpus(get_settings().corpus_dir)
    for d in docs:
        if key in d.all_drug_names():
            return {"name": key, "interactions": [d.id for d in docs if key in d.all_drug_names()]}
    raise HTTPException(status_code=404, detail="Drug not found")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_interactions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/api/test_interactions.py
git commit -m "feat(ml): RAG-backed /interactions/check-enhanced + corpus /drugs"
```

---

### Task 6: Retire the legacy FAISS app

**Files:**
- Delete: `ml/main.py`
- (Docs) note the new run command.

**Interfaces:**
- Consumes: nothing.
- Produces: `ml/main.py` removed; the ML service is launched with `uvicorn app.main:app` (from `ml/`, port 8000, matching Express `ML_BASE`).

- [ ] **Step 1: Confirm nothing imports the legacy module**

Run: `grep -rnE '\bimport main\b|from main import|ml\.main|\bmain:app\b' ml --include=*.py | grep -v venv`
Expected: no matches in `ml/app`, `ml/scripts`, `ml/tests` (only the soon-deleted `ml/main.py` itself may self-reference in its `uvicorn.run("main:app", ...)`).

- [ ] **Step 2: Delete the legacy app**

Run: `git rm ml/main.py`

- [ ] **Step 3: Run the full suite to verify nothing broke**

Run: `python -m pytest tests/unit tests/eval tests/api -q`
Expected: PASS (all offline tests, including the new `tests/api`). The eval/unit suites import from `app/`, not `main`, so deletion is safe.

- [ ] **Step 4: Verify the app boots**

Run: `python -c "from app.main import app; print(len(app.routes), 'routes')"`
Expected: prints a route count (≥ the 6 added). This confirms `app.main:app` is importable as the new uvicorn target.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(ml): delete legacy FAISS main.py; serve via app.main:app"
```

---

## Operator run book (provisioned machine)

The ML service now launches with the RAG app instead of the legacy file:

```bash
cd ml && source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`/health` and `/drugs` work with no live deps. `/query`, `/query/stream`, and `/interactions/check-enhanced` need the Pinecone index (Phase 1b/3) and Ollama running (`ollama pull llama3.1`) for real answers. Express's `ML_BASE` stays `http://localhost:8000`, so the existing proxy keeps resolving. Streaming end-to-end through Express + the frontend UI are Phase 4b/4c.

## Self-Review

- **Spec coverage:** `/health` + `/query` (Task 3), SSE `/query/stream` with the streamed==non-streamed invariant (Task 4), RAG-backed `/interactions/check-enhanced` + corpus `/drugs`/`/drugs/{name}` preserving contracts (Task 5), legacy deletion (Task 6); schemas + pure helpers (Tasks 1–2). Express (4b) and frontend (4c) are explicitly out of this plan.
- **Placeholder scan:** none — every code step has complete, runnable code.
- **Type consistency:** `QueryRequest{query, top_k}`, `Citation{source_doc_id, section, url}`, `QueryResponse{answer, citations}`, `InteractionResponse{severity, description, recommendation, sources, confidence, method}`, `EnhancedInteractionRequest{drugA, drugB, patientContext}`, `citations_from_chunks(chunks)`, `sse_frame(data, event=None)`, `build_interaction_response(chunks, answer)`, `get_retriever()`/`get_llm()`, and the `_tokens_from_sse` test helper against `sse_frame`'s exact `data: {..}\n\n` framing are used identically across tasks. `get_retriever` returns a `DenseRagPipeline` whose `.retrieve` matches the fakes' `retrieve(query, k)`.
- **Known simplification:** `sse_frame` puts the token on a single `data:` line; a token containing a raw newline would need multi-line `data:` framing. Real LLM tokens don't carry raw newlines mid-token in practice; noted for the live pass.
```
