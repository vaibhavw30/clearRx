# ClearRx — Drug-Interaction RAG (engineering debrief)

> **Status:** skeleton — offline metrics + narrative are final; numbers marked `__.__` are filled from the committed `eval/reports/*.json` after the live runs (see [`docs/live-runs-provisioning.md`](../docs/live-runs-provisioning.md)). Every `<!-- FILL -->` names the exact report field to copy.

A drug-drug interaction assistant, rebuilt from a fake "RAG" into a **real, evaluated** retrieval-augmented system. The point of this writeup is not that it retrieves — it's that **every change was measured against a labeled eval set before it was kept.**

## TL;DR (resume-grade)

Built an **LLM-as-judge evaluation harness first** (80-query-class labeled set split into retrieval and generation signals), then iterated a RAG pipeline against it: **BGE dense retrieval on Pinecone → empirically-chosen chunking → hybrid (BM25 + dense) with cross-encoder reranking → SSE-streamed, citation-grounded answers**, replacing a FAISS name-lookup + `gpt-4o-mini` prototype. Retrieval and generation are measured separately; the judge itself is calibrated against human labels.

- precision@5 **0.16 → __.__** &nbsp;·&nbsp; nDCG@5 **0.48 → __.__** &nbsp;·&nbsp; answer fact-coverage **0.0 → __.__** <!-- FILL from retrieval.json / dense.json -->
- Judge-vs-human agreement **__%** over the calibration set <!-- FILL dense.json calibration -->

## 1. The honest starting point

The inherited system used FAISS but was **not RAG**:

- `ml/main.py` embedded **6 hardcoded drugs** with `all-MiniLM-L6-v2` and did `drug_names.index(drug)` — a **name lookup**, not query→document retrieval. No corpus, no chunking, no query embedding.
- Generation was `gpt-4o-mini` with drug metadata pasted into the prompt; a rule-based path was the fallback.
- **Zero tests, zero evaluation** anywhere in the repo.

So "improve the RAG" first required *building* a RAG system and, before that, **something to measure it with.**

## 2. Methodology (what makes this more than plumbing)

- **Eval harness before any change.** A labeled query set with, per query: gold `expected_doc_ids` (ranking metrics), `expected_retrieval_topics` (coverage), `expected_answer_facts` (judge), and `must_not_say` (safety). An honest baseline was captured before touching retrieval.
- **Retrieval and generation measured separately.** Most RAG failures are *retrieval* failures; conflating them hides the cause. Ranking metrics (precision/recall/MRR/nDCG) are averaged only over queries that have a gold doc; negative "no-interaction" queries are graded on generation/safety only.
- **LLM-as-judge, calibrated.** A fact-by-fact rubric scores generated answers; the judge is **checked against a human-labeled subset** and reports agreement %, with a one-env-var escape hatch to a stronger judge if agreement is low. The measurement instrument is itself validated.
- **Baseline-first, empirical iteration.** Every phase re-runs the same eval and records the delta — chunking strategy and hybrid alpha are *chosen from data*, not asserted.
- **Discipline:** TDD throughout; every phase executed as reviewed, independently-tested tasks; heavy deps kept lazy so the whole unit/API suite runs offline (currently **109 ML + 5 Express + 6 frontend** tests green).

## 3. Results — baseline → final

Retrieval, over the labeled set (47 queries, 37 retrieval-gradable). All from committed `eval/reports/`.

| Metric | Keyword baseline | BGE dense | + best chunking | + hybrid + rerank |
|---|---|---|---|---|
| recall@5 | **0.5676** | __.__ <!-- dense.json recall_at_k --> | __.__ <!-- chunking.json winner --> | __.__ <!-- retrieval.json winner --> |
| precision@5 | **0.1586** | __.__ | __.__ | __.__ |
| nDCG@5 | **0.4822** | __.__ | __.__ | __.__ |
| MRR | **0.4536** | __.__ | __.__ | __.__ |
| retrieval coverage | **0.2838** | __.__ | __.__ | __.__ |

Generation / judge (from `dense.json` once the LLM is wired):

| Metric | Keyword baseline | BGE dense (Ollama + judge) |
|---|---|---|
| answer fact-coverage | **0.0** (no LLM) | __.__ <!-- dense.json fact_coverage --> |
| `must_not_say` violations | **__** <!-- baseline.json forbidden_violations --> | __ |
| judge-vs-human agreement | — | **__%** <!-- dense.json calibration --> |

> Baseline numbers are the **reset keyword baseline** captured after the corpus/eval-set expansion (§4.1) — deliberately harder than an 8-doc toy set, so the dense/hybrid lifts below are honest.

## 4. Per-change narrative + measured impact

Each subsection gets one line of "what changed" and one of "what it bought" (the number from the report).

### 4.1 Data foundation (Phase 1a)
Expanded the corpus 8 → **27** cited interaction monographs and the eval set 13 → **47** labeled queries (brand↔generic, dosage, and negative "no-interaction" cases so precision is testable) + **19** human calibration labels. Reset the keyword baseline as the new starting line.
**Impact:** turned a toy eval with no discriminating power into one where later deltas are trustworthy. <!-- narrative, no single number -->

### 4.2 Dense retrieval + real judge (Phase 1b)
BGE (`bge-large-en-v1.5`, 1024-d) embeddings on Pinecone (metadata + namespaces); local LLM (Ollama) for generation and the judge; judge calibrated against the human labels.
**Impact:** recall@5 __.__→__.__, fact-coverage 0.0→__.__, judge agreement __%. <!-- dense.json -->

### 4.3 Chunking, chosen empirically (Phase 2)
Compared fixed-size vs recursive (LangChain) vs semantic-breakpoint chunking, each re-indexed into its own namespace and retrieval-evaluated.
**Impact:** winner = **____** by nDCG@5 (recall tie-break); +__.__ nDCG@5 vs fixed. <!-- chunking.json winner + delta -->

### 4.4 Hybrid retrieval + reranking (Phase 3)
Added BM25 sparse (pinecone-text) + Pinecone hybrid with an **alpha sweep** to find the dense/sparse blend, then a two-stage rerank (local `bge-reranker` cross-encoder; Cohere pluggable).
**Impact:** best alpha = **____**; precision@5 __.__ (dense) → __.__ (hybrid) → __.__ (hybrid+rerank). <!-- retrieval.json -->

### 4.5 Streaming, end-to-end (Phase 4)
Replaced the legacy FAISS service with a thin FastAPI app: `/query` + `/query/stream` (SSE), plus RAG-backed `/interactions/check` preserving the dashboard contract (corpus severity mapped to the frontend's vocabulary). Streamed through Express to a live "Ask ClearRx" panel that renders tokens as they arrive with grounded citations.
**Impact:** tested invariant — the **streamed answer equals the non-streamed answer**; sub-second perceived latency; citations grounded in retrieved chunks (never the LLM). <!-- qualitative -->

## 5. Architecture

```
query
  → normalize/alias  → dense embed (BGE) [+ BM25 sparse]
  → Pinecone (dense | hybrid) top-50, metadata-filterable
  → cross-encoder rerank → top-5
  → grounded prompt → Ollama (stream) → SSE → browser
                                     ↳ citations from retrieved chunks
```
Every stage sits behind a `Protocol` with a fake, so the pipeline is swappable and the full suite runs offline. Shared helpers (`retrieval_experiment`, `chunks_from_matches`, `aggregate`) are extracted, not duplicated.

## 6. Limitations & honest notes

- Corpus is **curated original prose** (~27 interactions), not an exhaustive clinical database — this is an engineering/eval exercise; medical disclaimers stay.
- The live `/query` path uses **dense** retrieval; adopting the Phase 3 hybrid winner into the answer path is a documented manual follow-up.
- The BGE-BM25 `load` round-trip is currently unit-covered only; validate a real dump→load before adopting hybrid live.
- If a chunking/hybrid result comes back **flat** (within noise), that is reported as-is rather than manufactured — see the report files.

## 7. Reproduce

Offline (no keys): `cd ml && source venv/bin/activate && python -m pytest -q`.
Live metrics: follow [`docs/live-runs-provisioning.md`](../docs/live-runs-provisioning.md); the `eval/reports/*.{json,md}` it generates are the source of every number above.
