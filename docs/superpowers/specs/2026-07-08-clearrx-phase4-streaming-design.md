# ClearRx Phase 4 — Streaming RAG Serving Layer Design Spec

**Date:** 2026-07-08
**Status:** Approved (design), pending implementation plan
**Owner:** vaibhav
**Scope:** Stand up the real RAG serving layer end-to-end with token streaming — a new `app/main.py` FastAPI service exposing `/query` + `/query/stream` (SSE) over the RAG pipeline, retire the legacy FAISS `ml/main.py`, pass the stream through Express, and render it live in a new frontend "Ask ClearRx" panel. Corresponds to "Phase 4" of the rebuild (design spec §8 Step 5), expanded to fix the gap that no end-to-end RAG answer path exists yet.

---

## 1. Motivation & current-state findings

- The RAG pipeline (`DenseRagPipeline`/`HybridRerankRetriever` + `generator`) is fully built but **never served over HTTP**. The `app/` package has no FastAPI app — only `config.py`.
- The live FastAPI app is still the legacy `ml/main.py` (557 lines, FAISS name-lookup + `gpt-4o-mini`), exposing `/interactions/check-enhanced`, `/drugs`, `/drugs/{name}`, `/health`.
- The chain is Frontend (`frontend/src/services/api.ts`) → Express (`api/server.js`, `/api/...`) → legacy ML endpoints. The frontend is a patient/drug dashboard (`NewDashboard.tsx`, React + Vite + TanStack Query + shadcn/ui); there is **no free-text query UI**.
- `LLMClient.stream()` already exists on the Protocol (fallback yields the whole answer once; real Ollama streams chunks), so the streaming interface is ready.

Consequence: "add SSE streaming" is not a small change — there is no RAG answer path to stream. Phase 4 builds that path and streams it.

## 2. Decisions (this brainstorm)

| Concern | Decision |
|---|---|
| Reach | **Full stack** — ML service + Express passthrough + frontend UI. |
| Topology | **Replace legacy** — `app/main.py` becomes the single ML FastAPI service; `ml/main.py` is deleted. |
| Legacy endpoints | **Port existing behavior** of the dashboard-used routes (`/health`, `/interactions/check-enhanced`, `/drugs`, `/drugs/{name}`) onto `app/main.py` with response shapes **unchanged** — no regression. Interaction-check is NOT re-implemented on the RAG pipeline this phase (separate concern); `/query` is the new RAG surface. |
| Execution | **Layered, sequenced:** 4a ML service → 4b Express → 4c frontend, each independently testable. |
| Streaming | FastAPI `StreamingResponse` SSE (`text/event-stream`); token frames + a final citations event + `[DONE]`. |

## 3. Non-goals

- No re-backing of `/interactions/check-enhanced` on the RAG pipeline (behavior preserved as-is).
- No new clinical claims; the medical disclaimer stays in the answer path.
- No auth/session changes; the new query endpoint follows the existing unauthenticated dev pattern of the other routes.
- Live streaming answers require Ollama (provisioned machine); offline tests use a fake `LLMClient`.

## 4. Architecture

### 4a — ML service (`app/`)

| New / changed | Responsibility |
|---|---|
| `app/schemas.py` *(new)* | pydantic models: `QueryRequest{query: str, top_k: int = 5}`, `Citation{source_doc_id: str, section: str \| None, url: str \| None}`, `QueryResponse{answer: str, citations: list[Citation]}`, `HealthResponse`, and the ported legacy `InteractionRequest`/`InteractionResponse` shapes (verbatim from `ml/main.py`). |
| `app/deps.py` *(new)* | Builds the retriever (dense per settings; hybrid is a later swap) + `OllamaClient` from `Settings`; exposed as FastAPI dependencies (`get_retriever`, `get_llm`) so tests override them with fakes via `app.dependency_overrides`. |
| `app/main.py` *(new)* | Thin FastAPI app + routes only (no business logic): `GET /health`, `POST /query`, `POST /query/stream`, ported `POST /interactions/check-enhanced`, `GET /drugs`, `GET /drugs/{name}`. |
| `app/answer.py` *(new)* | Pure helpers: `citations_from_chunks(chunks) -> list[Citation]` (distinct source docs, first-seen order, url from metadata `source_url`) and `sse_frame(data, event=None) -> str`. Reuses `build_prompt`/`build_context` from `rag/pipeline.py`. |
| `ml/main.py` *(delete)* | Legacy FAISS app retired once its needed routes are ported. |

The route logic: `chunks = retriever.retrieve(query, top_k)` → `prompt = build_prompt(query, chunks)` → non-stream `llm.generate(prompt)`; stream `llm.stream(prompt)`. Citations derive from `chunks` via `citations_from_chunks`.

### 4b — Express (`api/server.js`)

- `POST /api/query` → JSON proxy to ML `/query`.
- `POST /api/query/stream` → `fetch` ML `/query/stream`, set `text/event-stream` headers, pipe `response.body` chunks straight to the client (`res.write` per chunk, `res.end` on completion).
- Existing routes keep pointing at `ML_BASE` (now the new app). First `Vitest` + `supertest` tests (the proxy currently has zero tests).

### 4c — Frontend (`frontend/src`)

- `services/sse.ts` *(new)*: a pure `parseSSE(chunk, state)` frame parser (buffers partial frames, emits `{token}` / `{citations}` / `{done}` events) — unit-tested in isolation.
- `services/api.ts`: `streamQuery(query, {onToken, onCitations, onDone, onError})` using `fetch` + `ReadableStream` reader feeding `parseSSE`.
- `components/AskClearRx.tsx` *(new)*: shadcn `Textarea`/`Button`/`Card` panel; submit → tokens render live into the answer area → citations list + medical disclaimer. Mounted on `NewDashboard`.

## 5. SSE contract & data flow

**`POST /query`** (non-stream): `retrieve → build_prompt → llm.generate` → `QueryResponse{answer, citations}`.

**`POST /query/stream`** (`StreamingResponse`, `media_type="text/event-stream"`):

```
retrieve → build_prompt →
  for tok in llm.stream(prompt):  yield  "data: {tok}\n\n"
  yield  "event: citations\ndata: {json}\n\n"
  yield  "data: [DONE]\n\n"
```

**Key invariant (tested):** the concatenation of the `data:` token payloads (excluding the `citations` event and `[DONE]`) equals the non-stream `answer`. A fake `LLMClient` whose `stream()` yields the same text `generate()` returns (in ≥2 chunks) proves the contract offline; real Ollama streams token chunks.

**Data path:** Frontend `streamQuery` → Express `POST /api/query/stream` → ML `POST /query/stream` → SSE frames flow back unbuffered through Express to the browser, parsed by `parseSSE`, tokens appended live.

## 6. Testing strategy

- **4a ML (offline, `TestClient` + fakes):** `app.dependency_overrides` injects a fake retriever (returns known chunks) + fake `LLMClient`. Tests: `/health`; `/query` returns `{answer, citations}` with citations derived from chunks; `/query/stream` emits SSE frames and **assembled token stream == `/query` answer**; no-chunk case → "no interaction information available" + empty citations; ported endpoints return the **same response shapes** as legacy (fixture-compared). `citations_from_chunks` and `sse_frame` unit-tested directly.
- **4a integration (`@pytest.mark.integration`, gated):** real `OllamaClient.stream` yields multiple chunks.
- **4b Express (`Vitest` + `supertest`, mock ML `fetch`):** `/api/query` JSON passthrough; `/api/query/stream` forwards SSE chunks in order with `text/event-stream` headers; ML error → surfaced status.
- **4c Frontend (`Vitest`):** `parseSSE` unit tests (single frame, split-across-chunks frame, citations event, `[DONE]`); a light render/smoke test of `AskClearRx` with a mocked `streamQuery`.
- **Regression:** all existing `ml/` unit + eval tests stay green; deleting `ml/main.py` must not break them (they import from `app/`, not the legacy file — verified during 4a).

## 7. Configuration

New env/settings as needed: `ml_port` (default 8000, unchanged so Express `ML_BASE` still resolves), `retriever_kind` (`dense` default; `hybrid` selectable later). No new secrets beyond Phase 1b/3. The new app runs via `uvicorn app.main:app` (replacing `python ml/main.py`); update any launch reference.

## 8. Risks & mitigations

- **Deleting legacy breaks the dashboard** → port the four used endpoints with byte-compatible response shapes first, fixture-compare in tests, delete `ml/main.py` only after `/health` + ported routes pass. If a legacy endpoint has behavior not worth porting, flag it rather than silently dropping.
- **SSE buffering** (proxy or server flushing) hides streaming → disable response buffering in Express passthrough (`res.flushHeaders()`, write chunks immediately); FastAPI `StreamingResponse` flushes per yield.
- **Streaming needs Ollama** → offline tests use a fake `LLMClient`; live run deferred to the provisioned machine.
- **Frontend partial SSE frames** → `parseSSE` buffers incomplete frames across reader chunks; covered by a split-frame unit test.
- **Citations correctness** → derived from retrieved chunks (not the LLM), so they are grounded regardless of generation.

## 9. Deliverables

1. `app/main.py` + `app/schemas.py` + `app/deps.py` + `app/answer.py`; ported legacy routes; `ml/main.py` deleted.
2. `/query` + `/query/stream` (SSE) with grounded citations; streamed==non-streamed invariant tested.
3. Express `/api/query` + `/api/query/stream` passthrough + first proxy tests.
4. Frontend `parseSSE` + `streamQuery` + `AskClearRx` panel rendering tokens live with citations + disclaimer.
5. Test suites (ML API/offline + gated integration; Express; frontend) with existing suites green.
6. Run book: launch via `uvicorn app.main:app`; live streaming validated on the Ollama machine.
