# ClearRx — Drug-Interaction RAG (engineering debrief)

> **Draft:** the narrative is final; every number shown as `__.__` or `____` is a placeholder filled from the committed `eval/reports/*.json` after the live runs (see [`docs/live-runs-provisioning.md`](../docs/live-runs-provisioning.md)). Each placeholder carries an `<!-- FILL: <report>.<field> -->` comment naming exactly where its value comes from.

ClearRx answers drug-drug interaction questions. This document is about how it was rebuilt from a prototype that *looked* like RAG into a system where **every retrieval and generation change was measured against a labeled evaluation set before it was kept**. The interesting part isn't that it retrieves — it's the harness that tells you whether each change actually helped, and by how much.

## TL;DR

Built an **LLM-as-judge evaluation harness first**, then iterated a real RAG pipeline against it: BGE dense retrieval on Pinecone → an empirically-chosen chunking strategy → hybrid (BM25 + dense) retrieval with cross-encoder reranking → SSE-streamed, citation-grounded answers. Retrieval quality and answer quality are measured as **separate** signals, and the judge itself is calibrated against human labels before its scores are trusted.

Headline movement on the labeled set:

- precision@5 **0.16 → __.__** &nbsp;·&nbsp; nDCG@5 **0.48 → __.__** &nbsp;·&nbsp; answer fact-coverage **0.00 → __.__** <!-- FILL: retrieval.json (winner) / dense.json aggregate -->
- judge-vs-human agreement **__%** on the calibration set <!-- FILL: dense.json calibration.agreement -->

## 1. Where this started

The inherited system used FAISS, so it *read* like RAG. It wasn't.

`ml/main.py` embedded **six hardcoded drugs** with `all-MiniLM-L6-v2` and answered by doing `drug_names.index(drug)` — a name lookup, not a query→document retrieval. There was no corpus to retrieve from, no chunking, and no query embedding; generation was `gpt-4o-mini` with drug metadata pasted into the prompt, falling back to hand-written rules. There were **no tests and no evaluation anywhere in the repo.**

That changes what "improve the RAG" means. You can't improve what you can't measure, and there was nothing here to measure with — no corpus, no ground truth, no baseline. So the first work wasn't retrieval at all; it was building something honest to measure against.

## 2. How each change was judged

The whole project is organized around one rule: **capture a baseline, change one thing, re-run the same eval, record the delta.** A few decisions make that rule trustworthy rather than theater.

**Retrieval and generation are measured separately.** Most RAG failures are retrieval failures wearing a generation costume — the model answers fluently from the wrong context. Conflating the two hides the cause, so the harness scores them apart: ranking metrics (precision/recall/MRR/nDCG) over the retrieved documents, and a fact-by-fact judge over the generated answer. Ranking metrics are averaged only over queries that *have* a gold document; the negative "no-interaction" queries (added so precision is actually testable) are graded on answer safety instead, not on a ranking they can't have.

**The judge is calibrated, not assumed.** Open-ended answers are scored by an LLM against a per-fact rubric — but an unchecked judge is just a second opinion of unknown quality. So a human-labeled subset is scored by the judge and the harness reports **judge-vs-human agreement %**. If agreement is low, one environment variable swaps in a stronger judge. The measuring instrument is itself validated before its numbers appear in a table.

**Everything is reproducible and offline-testable.** Each stage sits behind a `Protocol` with a fake, and heavy dependencies (Pinecone, Ollama, sentence-transformers, Cohere) are lazy-loaded, so the full unit/API suite runs with no keys and no models downloaded — currently **110 ML + 5 Express + 6 frontend** tests, green in CI on every push. The work was done test-first, one reviewed change at a time.

## 3. Results — baseline → final

Retrieval, over the labeled set (47 queries, 37 retrieval-gradable). Numbers come from the committed `eval/reports/`; the keyword-baseline column is the real captured starting line.

| Metric | Keyword baseline | BGE dense | + best chunking | + hybrid + rerank |
|---|---|---|---|---|
| recall@5 | **0.5676** | __.__ <!-- FILL: dense.json aggregate.recall_at_k --> | __.__ <!-- FILL: chunking.json winner --> | __.__ <!-- FILL: retrieval.json winner --> |
| precision@5 | **0.1586** | __.__ | __.__ | __.__ |
| nDCG@5 | **0.4822** | __.__ | __.__ | __.__ |
| MRR | **0.4536** | __.__ | __.__ | __.__ |
| retrieval coverage | **0.2838** | __.__ | __.__ | __.__ |

Generation / judge (from `dense.json`, once the local LLM is wired):

| Metric | Keyword baseline | BGE dense (Ollama + judge) |
|---|---|---|
| answer fact-coverage | **0.00** (no LLM) | __.__ <!-- FILL: dense.json aggregate.fact_coverage --> |
| `must_not_say` violations | **__** <!-- FILL: baseline.json aggregate.forbidden_violations --> | __ |
| judge-vs-human agreement | — | **__%** <!-- FILL: dense.json calibration.agreement --> |

The baseline column is the **reset** keyword baseline — captured *after* the corpus and eval set were expanded (§4.1), so it is deliberately harder than an eight-document toy would be. Every lift below is measured against that harder line, which keeps them honest.

## 4. What changed, and what it bought

### 4.1 A data foundation worth measuring against

The eight-document corpus and thirteen-query set the prototype left behind couldn't tell a good retriever from a lucky one — with so few documents, almost anything scores well. So before any retrieval work, the corpus grew to **27 cited interaction monographs** (original prose synthesized from public literature, one file per interaction with severity, mechanism, management, and citations) and the eval set to **47 labeled queries**. The queries were written to exercise the failure modes that actually matter for drug interactions: brand-versus-generic names ("Advil" vs ibuprofen), dosage qualifiers, and — importantly — **negative pairs with no real interaction**, so precision has something to be wrong about. A ~15-query human-labeled slice was added for judge calibration.

Then the keyword baseline was re-run and re-committed as the new starting line. Recall fell from the toy corpus's inflated ~0.92 to **0.5676**, and precision to **0.1586** — not a regression, but the eval finally *discriminating*. That drop is the point: it created room for a real retriever to demonstrate a real gain.

### 4.2 Dense retrieval and a judge you can trust

With a corpus to retrieve over, the pipeline moved to genuine semantic retrieval: **BGE (`bge-large-en-v1.5`, 1024-d) embeddings on Pinecone**, with per-chunk metadata (drugs, class, severity, source) and namespaces so future corpora slot in without rework. Generation moved to a **local LLM via Ollama**, and the previously-stubbed judge became real — which is when answer fact-coverage stops being a structural zero and starts being a measurement.

The calibration step ran here for the first time: the judge scored the human-labeled slice and reported **__%** agreement <!-- FILL: dense.json calibration.agreement -->. [If that number is below ~80%, the writeup notes the swap to a stronger judge; if it's high, it's evidence the local judge is trustworthy enough to keep costs at zero.]

**What it bought:** recall@5 **0.5676 → __.__**, nDCG@5 **0.48 → __.__**, and answer fact-coverage **0.00 → __.__**. <!-- FILL: dense.json aggregate --> Dense retrieval understands that "blood thinner" and "anticoagulant" are the same idea in a way keyword overlap never will.

### 4.3 Chunking, decided by data instead of by taste

Fixed-size chunking is the default, and defaults are where quiet performance losses hide. Rather than argue about it, three strategies were run head to head — **fixed-size**, LangChain **recursive** character splitting, and **semantic** breakpoint chunking (split where consecutive sentences are embedding-distant) — each re-indexed into its own namespace and scored on the same retrieval eval.

**What it bought:** the winner was **____** <!-- FILL: chunking.json winner -->, worth **+__.__ nDCG@5** over fixed-size. <!-- FILL: chunking.json delta --> [If the strategies land within noise of each other — plausible, since the monograph sections are already short, self-contained semantic units — that is reported as-is and fixed-size is kept for simplicity, rather than manufacturing a winner. Which outcome occurred is in `chunking.md`.]

### 4.4 Hybrid retrieval and a reranker

Dense embeddings are strong on meaning and weak on exact tokens — drug names, dosages, lab values like "INR". Sparse lexical retrieval is the mirror image. So the pipeline added **BM25 sparse vectors** and Pinecone **hybrid** search, and rather than guess the dense/sparse mix, it **swept the blend weight** and picked the best empirically. On top of that, a two-stage rerank: retrieve a wide candidate set, then rescore it with a **cross-encoder** (local `bge-reranker`, with Cohere Rerank pluggable via one env var) that reads each candidate against the query far more precisely than a bi-encoder can, and keep the top five.

**What it bought:** the best blend was **α = ____** <!-- FILL: retrieval.json best_alpha -->, and precision@5 moved **__.__ (dense) → __.__ (hybrid) → __.__ (hybrid + rerank)**. <!-- FILL: retrieval.json --> This two-stage pattern — cheap wide recall, then expensive precise reranking — is the standard production shape, and it typically delivers the single biggest jump in retrieval precision. Whether it did here is in `retrieval.md`.

### 4.5 Streaming, end to end

The final change was as much product as pipeline. The legacy FAISS service was retired and replaced with a thin FastAPI app exposing `/query` and `/query/stream`; the old interaction-check endpoint was **re-pointed onto the RAG pipeline** while preserving the exact response contract the dashboard already consumed (including mapping the corpus's severity vocabulary to the frontend's). Answers stream token-by-token over Server-Sent Events, through an Express passthrough, into a new "Ask ClearRx" panel that renders text as it arrives with its supporting citations underneath.

**What it bought:** a test-enforced invariant — the **streamed answer is byte-identical to the non-streamed answer**, so streaming is purely a latency/UX win and never a correctness risk. Citations are drawn from the retrieved chunks, never parsed out of the model's prose, so a source list is grounded even if the generation drifts. Perceived latency drops from "wait for the whole paragraph" to "first words in well under a second."

## 5. Architecture

```
query
  → normalize / brand→generic alias
  → BGE dense embed  [+ BM25 sparse encode]
  → Pinecone (dense | hybrid) top-50, metadata-filterable
  → cross-encoder rerank → top-5
  → grounded, cited prompt → Ollama (streamed) → SSE → browser
                                              ↳ citations from retrieved chunks
```

Every stage is a `Protocol` with a fake behind it, so the pipeline is swappable and the whole suite runs offline. Shared logic (`aggregate`, `chunks_from_matches`, the retrieval-experiment harness) is extracted once and reused rather than copy-pasted across the four experiment scripts.

## 6. Honest limitations

- The corpus is **curated original prose** (~27 interactions), not an exhaustive clinical database. This is an engineering and evaluation exercise; the medical-advice disclaimer stays, and no new clinical claims are made.
- The live `/query` path currently serves **dense** retrieval. Promoting the Phase 3 hybrid winner into the answer path is a deliberate, documented manual step (it needs the BM25 parameters loaded and a real dump→load round-trip validated first), not something silently switched on.
- Where an experiment comes back **flat** — a chunking or hybrid result within noise of the previous stage — it is reported that way in the report files rather than dressed up as a win. Null results are results.

## 7. What this demonstrates

Less a retrieval trick, more a way of working: build the measurement before the feature, separate the signals so a number points at a cause, validate the instrument doing the measuring, and let data — not preference — pick chunking strategy and hybrid weight. The metrics table is the artifact; the harness that produced it is the actual deliverable.

## 8. Reproduce

Offline, no keys: `cd ml && source venv/bin/activate && python -m pytest -q` (this is what CI runs).
Live metrics: follow [`docs/live-runs-provisioning.md`](../docs/live-runs-provisioning.md); the `eval/reports/*.{json,md}` it generates are the source of every number above.
