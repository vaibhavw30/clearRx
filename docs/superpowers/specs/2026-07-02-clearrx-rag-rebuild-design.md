# ClearRx RAG Rebuild — Design Spec

**Date:** 2026-07-02
**Status:** Approved (design), pending implementation plan
**Owner:** vaibhav
**Scope:** Rebuild the ClearRx drug-drug interaction assistant as a real, evaluated RAG system inside the `ml/` service. Corresponds to "Phase 2" of the broader improvement doc, expanded to fix a foundational gap the doc assumed away (there is currently no real corpus and no evaluation).

---

## 1. Motivation & current-state findings

The current system is **not** a RAG system despite using FAISS:

- `ml/main.py` (557 lines) embeds **6 hardcoded drugs** with `all-MiniLM-L6-v2` (384-dim) into a FAISS `IndexFlatIP`.
- "Retrieval" is `drug_names.index(drug_a)` — a **direct name lookup**, not a semantic query→document retrieval. There is no query embedding, no chunking, and no corpus of documents.
- Generation is OpenAI `gpt-4o-mini` with drug metadata pasted into the prompt; a rule-based path is the fallback.
- A Supabase `drug_docs` table (`content`, `embedding`, `section`, `source_url`) and a `search_drug_docs` pgvector RPC **exist but are unused** by the ML service.
- **There are zero tests and zero evaluation anywhere in the repo.**

Consequences for the plan:

1. The improvement doc's Steps 3–4 (chunking strategies, hybrid search, reranking) assume a real document corpus to retrieve over. There is essentially nothing to chunk today. **Building a real corpus is the true Step 0.**
2. "You can't improve what you can't measure" is doubly true here — there is no baseline, no test harness, and no ground truth.
3. This is a portfolio/learning project (the source doc is organized around resume bullets), so the design biases toward canonical, demonstrable production patterns over shortcuts.

## 2. Goals & non-goals

### Goals
- A real RAG pipeline: query → retrieve over a document corpus → rerank → grounded, cited, streamed answer.
- An **evaluation harness built first**, measuring retrieval and generation as **separate** signals, with an honest baseline captured before any change.
- Empirically-chosen chunking, hybrid dense+sparse retrieval, and two-stage reranking, each validated by re-running the eval and recording the lift.
- Thorough tests (TDD) at unit, integration, API, and eval-regression levels.
- A README debrief documenting every change and its measured impact.

### Non-goals
- No change to the React UI beyond consuming streamed responses.
- No new clinical claims of correctness — this is an engineering/eval exercise, and every answer is grounded in and cited to the curated corpus. Medical-advice disclaimers stay.
- No migration off Supabase for patient/prescription data. Pinecone is added **only** as the RAG vector store.
- Not building an ingestion pipeline for FDA/DrugBank in this phase (curated corpus chosen); the vector store is namespaced so a future FDA corpus can be added without rework.

## 3. Chosen stack (decisions)

| Concern | Decision | Notes |
|---|---|---|
| Corpus | **Curated interaction monographs** | Original prose synthesized from public literature (avoids DrugBank licensing); ~40–50 drugs, ~60–80 interactions. |
| Embeddings | **Open-source local** (BGE, e.g. `bge-large-en-v1.5`, 1024-dim; E5 as alternative) | No API cost; dimension pinned to the Pinecone index. |
| Generation | **Open-source local LLM via Ollama** (e.g. Llama 3.1 / Qwen 2.5) | Pluggable behind `LLMClient`. |
| LLM-as-judge | **Local by default, pluggable, calibrated** | See §7. One env var switches to a stronger judge; judge-agreement is measured and reported. |
| Vector store | **Pinecone** (serverless, free tier) | Namespaces + metadata filtering + native hybrid. |
| Reranker | **Cohere Rerank** (`rerank-english-v3.0`) | Paid API key; behind `Reranker` interface. |
| Sparse | **BM25 via `pinecone-text`** | For hybrid retrieval. |
| API | FastAPI (ML) + Express proxy (unchanged topology) | Streaming added end-to-end. |

## 4. Architecture

Restructure `ml/` from one file into a testable package. Every stage sits behind an interface (Python `Protocol`) so it is swappable and unit-testable with a fake.

```
ml/
  app/
    main.py            # FastAPI app + routes (thin; no business logic)
    config.py          # pydantic-settings; env, keys, model names, dims
    schemas.py         # pydantic request/response models
    rag/
      corpus.py        # load + validate corpus monographs -> Document objects
      chunking.py      # Chunker protocol + Fixed / Recursive / Semantic impls
      embeddings.py    # Embedder protocol + BGE/E5 impl; exposes .dimension
      sparse.py        # SparseEncoder protocol + BM25 (pinecone-text) impl
      vectorstore.py   # VectorStore protocol + Pinecone impl (upsert/query/hybrid)
      retriever.py     # Reranker protocol + Cohere impl; dense/hybrid + rerank stages
      generator.py     # LLMClient protocol + Ollama impl; .generate and .stream
      pipeline.py      # orchestrates resolve->retrieve->rerank->generate
      aliases.py       # brand<->generic drug-name resolution
    eval/
      dataset.py       # load + validate eval queries JSON
      metrics.py       # coverage, precision@k, recall@k, MRR, nDCG, latency, cost
      judge.py         # Judge protocol + LLM-as-judge impl; fact coverage; must_not_say
      runner.py        # run eval over a pipeline; emit json + markdown report
      baseline.py      # adapter wrapping the CURRENT name-lookup system
    ingest/
      build_index.py   # CLI: corpus -> chunk -> embed -> sparse -> upsert to Pinecone
  data/
    corpus/            # curated interaction monographs (one file per interaction)
    eval/queries.json  # 80+ labeled queries
    eval/human_labels.json  # ~15-query human-labeled subset for judge calibration
  eval/reports/        # timestamped json + markdown experiment reports (committed)
  tests/
    unit/              # per-module, fakes only, fast
    integration/       # real Pinecone/Cohere/Ollama, @pytest.mark.integration
    api/               # FastAPI endpoint tests incl. streaming
    eval/              # golden-report regression thresholds
  requirements.txt
  pyproject.toml       # pytest config, markers, ruff
```

### Interfaces (contracts)

```python
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...

class Embedder(Protocol):
    dimension: int
    def embed(self, texts: list[str]) -> np.ndarray: ...   # (n, dimension)

class SparseEncoder(Protocol):
    def fit(self, corpus: list[str]) -> None: ...
    def encode_documents(self, texts: list[str]) -> list[SparseVector]: ...
    def encode_query(self, text: str) -> SparseVector: ...

class VectorStore(Protocol):
    def upsert(self, records: list[Record], namespace: str) -> None: ...
    def query(self, dense, sparse=None, top_k=10, flt=None, namespace=str) -> list[Match]: ...

class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str], top_n: int) -> list[RankedDoc]: ...

class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...

class Judge(Protocol):
    def score_facts(self, answer: str, facts: list[str]) -> list[bool]: ...
    def check_forbidden(self, answer: str, must_not_say: list[str]) -> list[bool]: ...
```

## 5. Data models

### Corpus monograph (`ml/data/corpus/*.json`)
Original prose, publicly-sourced, cited. Sections provide chunkable text; metadata attaches to every derived chunk.

```json
{
  "id": "int_warfarin_ibuprofen",
  "drug_a": "warfarin", "drug_b": "ibuprofen",
  "drug_a_aliases": ["coumadin"],
  "drug_b_aliases": ["advil", "motrin", "nurofen"],
  "drug_class_a": "anticoagulant", "drug_class_b": "nsaid",
  "severity": "high",
  "sections": {
    "summary": "...",
    "mechanism": "...",
    "clinical_effects": "...",
    "management": "...",
    "monitoring": "..."
  },
  "evidence": [{"citation": "…", "url": "https://…"}]
}
```

Each `Chunk` carries: `chunk_text`, `source_doc_id`, `section`, `chunk_index`, and filterable metadata (`drugs_mentioned`, `drug_class`, `severity`, `source_url`).

### Eval query (`ml/data/eval/queries.json`)
Extends the doc's schema so retrieval can be measured with ranking metrics and answers checked for safety.

```json
{
  "id": "q001",
  "query": "Can I take ibuprofen with my warfarin prescription?",
  "query_type": "interaction",
  "expected_doc_ids": ["int_warfarin_ibuprofen"],
  "expected_retrieval_topics": ["ibuprofen-warfarin interaction", "NSAID anticoagulant"],
  "expected_answer_facts": [
    "Increased bleeding risk",
    "NSAIDs reduce platelet function",
    "Monitor INR if combined"
  ],
  "must_not_say": ["safe to combine", "no interaction"],
  "severity": "high"
}
```

- `expected_doc_ids` → precision@k, recall@k, MRR, nDCG (real ranking metrics, not just substring coverage).
- `expected_retrieval_topics` → the doc's substring coverage metric (kept for continuity).
- `expected_answer_facts` → judge fact-coverage.
- `must_not_say` → safety guard against dangerous hallucinations (false negatives are the harmful failure mode here).

## 6. Query pipeline

1. **Normalize + resolve** the query's drug names brand↔generic via `aliases.py` (targets the doc's synonym/brand-vs-generic failure modes).
2. **Dense embed** (BGE) + **sparse encode** (BM25) the query.
3. **Hybrid retrieve** top-50 from Pinecone, with optional metadata filter (`severity`, `drug_class`), from the `curated` namespace.
4. **Cohere rerank** → top-5.
5. **Build context** from reranked chunks; **generate** with the local LLM.
6. **Stream** tokens over SSE; return the answer plus **citations** (source doc ids / urls).

Dense-only retrieval is implemented first (Phase 1) to prove parity with the FAISS baseline before hybrid/rerank are layered on.

## 7. Evaluation harness (built first)

- **Baseline adapter** (`eval/baseline.py`) wraps the *current* name-lookup + `gpt-4o-mini`/rule-based system so the same runner can score it. This is the honest starting line; captured before any change.
- **Metrics** (`eval/metrics.py`): retrieval coverage; precision@k / recall@k; MRR@10; nDCG@10; answer fact-coverage; `must_not_say` violation count; latency p50/p95; token/cost estimate. Retrieval and generation are reported **separately**.
- **Judge** (`eval/judge.py`): LLM-as-judge with a fact-by-fact rubric returning one boolean per expected fact (structured output, retried on parse failure). **Pluggable** — local model by default; a single env var (`JUDGE_MODEL`/`JUDGE_PROVIDER`) switches to a stronger judge.
- **Judge calibration**: a ~15-query human-labeled subset (`data/eval/human_labels.json`) is scored by the judge; the runner reports **judge-vs-human agreement %**. If agreement is low, the escape hatch to a stronger judge is used. This makes the measurement instrument itself trustworthy.
- **Reports**: every run writes a timestamped `json` + `markdown` table to `ml/eval/reports/`. These are committed and become the README results table (baseline → each improvement → final).

## 8. The improvement steps (mapped to the doc)

- **Step 1 — Eval harness first (Phase 0):** corpus + queries + baseline + metrics + judge + calibration.
- **Step 2 — Pinecone (Phase 1):** index creation at the embedder's dimension; upsert with metadata; namespaces; dense-only query. Re-run eval → expect parity with FAISS baseline (goal: learn the production interface, not change quality).
- **Step 3 — Chunking (Phase 2):** three `Chunker` impls (fixed-512, `RecursiveCharacterTextSplitter`, semantic-breakpoint). Re-index and eval each; choose empirically; document which and why with the coverage delta.
- **Step 4 — Hybrid + rerank (Phase 3):** BM25 sparse + Pinecone hybrid; Cohere rerank as a top-50→top-5 stage. Eval dense-only vs hybrid vs hybrid+rerank; report precision@5 lift.
- **Step 5 — Streaming (Phase 4):** FastAPI `StreamingResponse` (SSE) → Express passthrough → frontend `fetch` ReadableStream. Verify streamed answer equals non-streamed.
- **Phase 5 — Debrief:** README writeup with the full metrics table and per-change impact.

## 9. Testing strategy (TDD, thorough)

- **Unit** (`tests/unit`, fakes only, fast, run in CI always): metrics math against known fixtures (hand-computed precision/recall/MRR/nDCG); chunker boundary behavior; alias resolution; judge output parsing; sparse encoder determinism; pipeline orchestration with all stages mocked; corpus schema validation; eval dataset validation.
- **Integration** (`tests/integration`, `@pytest.mark.integration`, skipped without keys): real Pinecone upsert/query round-trip; real Cohere rerank ordering sanity; real Ollama generate/stream; real embedder dimension.
- **API** (`tests/api`, httpx against FastAPI): request validation, non-streaming endpoint, **streaming endpoint** asserts chunked SSE frames and that the assembled stream equals the non-streamed answer.
- **Express** (`api/`, Vitest + supertest — currently zero tests): proxy passthrough and streaming passthrough.
- **Eval regression** (`tests/eval`): a committed golden report defines minimum thresholds; a test fails if metrics regress below them.
- **CI** (GitHub Actions): unit + fake-provider eval on every push; integration gated on repository secrets.

## 10. Configuration & secrets

New env vars (added to `.env` / `.env.example`): `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_NAMESPACE`, `COHERE_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `OLLAMA_BASE_URL`, `GEN_MODEL`, `JUDGE_PROVIDER`, `JUDGE_MODEL`. `config.py` (pydantic-settings) centralizes and validates them; missing integration keys degrade to skipped integration tests, never silent wrong answers.

## 11. Risks & mitigations

- **Local judge unreliable** → calibration % + one-env-var escape hatch to a stronger judge (§7).
- **Curated corpus too small to differentiate chunking strategies** → target ≥60 interactions with multi-paragraph sections; if signal is flat, note it honestly in the debrief rather than manufacturing a delta.
- **Pinecone free-tier / hybrid metric constraints** (hybrid needs dotproduct-metric index) → pin index config in `build_index.py`; document it.
- **Cohere/Pinecone availability in CI** → all external calls behind interfaces + markers; CI runs on fakes.
- **Scope creep back into FDA ingestion** → explicitly deferred; namespace reserved.

## 12. Deliverables

1. Restructured, tested `ml/app` package with the RAG pipeline.
2. Curated corpus (~40–50 drugs / ~60–80 interactions) + 80+ labeled eval queries + ~15 human labels.
3. Eval harness with committed baseline→final reports.
4. Pinecone-backed retrieval with metadata filtering + namespaces.
5. Empirically-selected chunking; hybrid + Cohere rerank; end-to-end streaming.
6. Test suites (unit/integration/api/eval) + Express tests + CI.
7. README debrief with the metrics table and per-change impact.
