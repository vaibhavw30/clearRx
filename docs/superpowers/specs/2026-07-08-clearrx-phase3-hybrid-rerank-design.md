# ClearRx Phase 3 — Hybrid Retrieval + Reranking Design Spec

**Date:** 2026-07-08
**Status:** Approved (design), pending implementation plan
**Owner:** vaibhav
**Scope:** Add BM25 sparse retrieval + Pinecone hybrid search and a two-stage rerank (local cross-encoder by default, Cohere as a pluggable hosted option) to the ClearRx RAG service, plus a retrieval-only experiment harness that empirically compares dense vs hybrid (alpha-swept) vs hybrid+rerank. Corresponds to "Phase 3" of the broader rebuild (design spec §8 Step 4). Builds on Phase 1b (dense BGE + Pinecone + Ollama) and Phase 2 (chunking experiments + shared retrieval-aggregation helpers).

---

## 1. Motivation

Dense embeddings alone miss exact-match retrieval cues that matter for drug interactions — brand/generic names, dosage numbers, specific lab terms (e.g. "INR"). Sparse lexical retrieval (BM25) complements dense semantics, and a cross-encoder reranker gives a cheap-recall → expensive-precision two-stage pipeline that is the standard production pattern. Phase 3 adds both and measures the lift honestly against the Phase 1b/2 dense baseline.

## 2. Decisions (this brainstorm)

| Concern | Decision | Rationale |
|---|---|---|
| Reranker | **Pluggable**: local `bge-reranker-base` cross-encoder default; Cohere `rerank-english-v3.0` via `RERANK_PROVIDER` env | Mirrors the existing pluggable-judge pattern; free by default, keeps the zero-cost local story, still demonstrates the recognizable Cohere production tool. |
| Experiment | **Retrieval-only + hybrid alpha sweep** | Runs on the user's machine (no Ollama); rerank's value is a retrieval-precision signal. A fixed-alpha hybrid is an under-tuned comparison, so sweep alpha to find the real best blend before reranking — the reported lift is honest, not arbitrary. |
| Sparse | **BM25 via `pinecone-text`** | Spec-chosen; `BM25Encoder` fits corpus stats at build time, params persisted so query-time encoding matches. |
| Hybrid store | **Pinecone native hybrid** on the existing `dotproduct` index | The Phase 1b index metric is already `dotproduct`, which hybrid requires — no blocking reindex. |
| Live adoption | **Deferred to a manual follow-up** | Consistent with Phase 2: build components + experiment; wiring the winner into the live answer path is a documented manual step, not auto-wired. |

## 3. Non-goals

- No end-to-end generation/judge eval in the harness (retrieval-only chosen; needs Ollama, adds friction, little added signal about the retrieval change). Full end-to-end and the README debrief are Phase 5.
- No auto-rewiring of `run_dense.py` / `DenseRagPipeline` to hybrid+rerank — manual follow-up.
- No change to the corpus, eval queries, chunking strategies, or the dense/Phase-2 code paths (sparse is strictly additive; existing behavior is preserved).

## 4. Architecture — components & interfaces

Every stage sits behind a `Protocol` and is unit-tested with a fake; heavy deps (`pinecone-text`, `cohere`, sentence-transformers `CrossEncoder`) are lazy-imported inside methods so the offline unit suite needs none of them installed.

| New / changed | Responsibility |
|---|---|
| `app/rag/sparse.py` *(new)* | `SparseEncoder` Protocol + `BM25SparseEncoder` wrapping pinecone-text `BM25Encoder`. `.fit(corpus_texts)` at build time; params dumped to `ml/data/bm25_params.json` (committed); `encode_documents(texts)` / `encode_query(text)` return Pinecone sparse dicts `{indices, values}`. |
| `app/rag/rerank.py` *(new)* | `Reranker` Protocol + `LocalReranker` (sentence-transformers `CrossEncoder`, `bge-reranker-base`, lazy) + `CohereReranker` (lazy `cohere`) + `build_reranker(settings)` factory — local default, Cohere via `RERANK_PROVIDER`. `rerank(query, docs, top_n) -> list[int]` (indices into `docs`, best first). |
| `app/rag/vectorstore.py` *(extend)* | `Record` gains `sparse_values: Optional[dict] = None`; `query` gains `sparse: Optional[dict] = None`. **Both default `None` → pure-dense behavior unchanged**, so `DenseRagPipeline` and the Phase 2 harness keep working untouched. `upsert` includes `sparse_values` when present. |
| `app/rag/hybrid.py` *(new)* | `HybridRerankRetriever` composing embedder + sparse encoder + store + (optional) reranker. `retrieve(query, k)`: dense embed + sparse encode → convex alpha-scale → store hybrid query top-50 → optional rerank → top-k `Chunk`s. `RagPipeline`-compatible `.retrieve`. |
| `app/ingest/build_index.py` *(extend)* | Add a hybrid record builder: fit BM25 on chunk texts, dump params, attach `sparse_values` to records, upsert to a `hybrid` namespace. Existing dense `build_records` unchanged. |
| `scripts/run_retrieval_experiment.py` *(new)* | Harness: dense baseline + hybrid alpha sweep + hybrid(best-alpha)+rerank; reuses `evaluate_retrieval` from the Phase 2 harness (or the shared aggregate helpers); writes `eval/reports/retrieval.{json,md}` + the winning config. |
| `app/config.py` *(extend)* | `rerank_provider="local"`, `rerank_model_local="BAAI/bge-reranker-base"`, `cohere_api_key=""`, `cohere_rerank_model="rerank-english-v3.0"`, `hybrid_alphas=[0.0,0.25,0.5,0.75,1.0]`, `hybrid_top_k=50`, `bm25_params_path`. Deps added to `requirements.txt`: `pinecone-text`, `cohere`. |

### Interface sketches

```python
class SparseEncoder(Protocol):
    def fit(self, corpus_texts: list[str]) -> None: ...
    def encode_documents(self, texts: list[str]) -> list[dict]: ...   # [{indices, values}]
    def encode_query(self, text: str) -> dict: ...
    def dump(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]: ...  # indices, best first

# VectorStore.query extended (sparse optional, back-compatible)
def query(self, dense, top_k, flt, namespace, sparse: Optional[dict] = None) -> list[Match]: ...
```

## 5. Data flow

**Index-build (offline, provisioned machine):**
```
corpus → chunk → per chunk: dense = BGE.embed(text); sparse = bm25.encode_documents(text)
       → Record{id, values=dense, sparse_values=sparse, metadata}
       → PineconeStore.upsert(records, namespace="hybrid")
       → bm25.dump(ml/data/bm25_params.json)
(bm25.fit(all chunk texts) runs once before encoding documents)
```

**Query-time (`HybridRerankRetriever.retrieve`):**
```
dense  = embedder.embed_query(q)
sparse = bm25.encode_query(q)                       # loaded from committed params
d', s' = convex_scale(dense, sparse, alpha)         # alpha*dense, (1-alpha)*sparse
cands  = store.query(dense=d', sparse=s', top_k=50, namespace="hybrid")
idxs   = reranker.rerank(q, [c.text for c in cands], top_n=k)   # skipped if rerank disabled
→ return [cands_as_chunks[i] for i in idxs][:k]
```

**Experiment (`run_retrieval_experiment.py`):**
```
dense-only:          evaluate_retrieval(dense_retriever, queries, k=5)
hybrid @ each alpha: for a in hybrid_alphas: evaluate_retrieval(hybrid(a, rerank=off))
best_alpha = argmax nDCG@5 over the sweep (recall@5 tie-break)
hybrid+rerank:       evaluate_retrieval(hybrid(best_alpha, rerank=on))
→ compare table + winner → eval/reports/retrieval.{json,md}
```

## 6. Testing strategy (mirrors Phase 1b/2)

- **Unit (offline, fakes only, no heavy deps imported):**
  - `BM25SparseEncoder` with a fake pinecone-text encoder — fit/encode/dump/load round-trip.
  - `LocalReranker` with a fake `CrossEncoder` — reorders by score, respects `top_n`; `build_reranker` factory selection (local vs cohere by env).
  - `HybridRerankRetriever` with fake embedder/sparse/store/reranker — convex scaling, top-50→rerank→top-k wiring, rerank-off path.
  - Harness pure functions (`pick_best_alpha`, compare-table builder) with fakes.
  - Regression: existing dense + Phase 2 tests stay green (sparse additive, defaults `None`).
- **Integration (`@pytest.mark.integration`, gated on keys/services):** real Pinecone hybrid upsert/query round-trip; real Cohere rerank ordering sanity; real local `CrossEncoder` load + score.

## 7. Prerequisites (experiment run only, provisioned machine)

Phase 1b already run (deps, `PINECONE_API_KEY`). Then: `pip install -r requirements.txt` (pulls `pinecone-text`, `cohere`), rebuild the `hybrid` namespace via the hybrid index builder, then `python -m scripts.run_retrieval_experiment`. `bge-reranker-base` downloads (~1GB) on first `CrossEncoder` load; runs on CPU. Cohere path needs `COHERE_API_KEY` + `RERANK_PROVIDER=cohere` (optional).

## 8. Risks & mitigations

- **BM25 fit/query mismatch** → params persisted to a committed file and loaded at query time; a unit test covers the dump/load round-trip.
- **Hybrid requires dotproduct index** → already the Phase 1b metric; documented in the builder.
- **Local reranker slow/heavy** → `bge-reranker-base` (small) on CPU is adequate for ~47 queries × 50 candidates; Cohere hatch available if needed.
- **Flat result** (hybrid/rerank within noise) → document honestly rather than manufacturing a winner, as in Phase 2 (design spec §11).
- **Offline suite must not require heavy deps** → all of pinecone-text/cohere/CrossEncoder lazy-imported; unit tests use fakes.

## 9. Deliverables

1. `sparse.py` (BM25) + `rerank.py` (pluggable local/Cohere) + `hybrid.py` (composed retriever), all behind Protocols with fakes.
2. `vectorstore.py` extended for sparse (back-compatible).
3. Hybrid index builder + committed BM25 params.
4. `run_retrieval_experiment.py` retrieval-only harness (dense / alpha-swept hybrid / hybrid+rerank) → committed `retrieval.{json,md}` after the live run.
5. Unit + integration tests; existing suites stay green.
6. Manual run book for the live experiment and for adopting the winning config into the answer pipeline (deferred follow-up).
