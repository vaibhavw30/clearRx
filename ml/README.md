# ClearRx — Drug-Interaction RAG (engineering debrief)

ClearRx answers drug-drug interaction questions. This document is about how it was rebuilt from a prototype that *looked* like RAG into a system where **every retrieval and generation change was measured against a labeled evaluation set before it was kept**. The interesting part isn't that it retrieves — it's the harness that tells you whether each change actually helped, and by how much, including the changes that **didn't** help.

All numbers below come from the committed `eval/reports/*.json`, produced on a local run (BGE embeddings, Pinecone, Ollama `llama3.1` generation, Ollama `qwen2.5:7b` judge). Reproduce via [`../docs/live-runs-provisioning.md`](../docs/live-runs-provisioning.md).

## TL;DR

Built an **LLM-as-judge evaluation harness first**, then iterated a real RAG pipeline against it and measured retrieval and generation as **separate** signals. Dense retrieval was a large, unambiguous win; recursive chunking and a well-tuned hybrid blend each added a bit more; a cross-encoder reranker **hurt and was dropped**. The judge was calibrated against human labels before its scores were trusted.

Headline, keyword baseline → best configuration, on the labeled set (47 queries, 37 retrieval-gradable):

- recall@5 **0.57 → 0.92** · precision@5 **0.16 → 0.36** · nDCG@5 **0.48 → 0.85**
- answer fact-coverage **0.00 → 0.57**, from a judge validated at **80.4%** human agreement

## 1. Where this started

The inherited system used FAISS, so it *read* like RAG. It wasn't.

`ml/main.py` embedded **six hardcoded drugs** with `all-MiniLM-L6-v2` and answered by doing `drug_names.index(drug)` — a name lookup, not a query→document retrieval. There was no corpus to retrieve from, no chunking, no query embedding; generation was `gpt-4o-mini` with drug metadata pasted into the prompt, falling back to hand-written rules. There were **no tests and no evaluation anywhere in the repo.**

That changes what "improve the RAG" means. You can't improve what you can't measure, and there was nothing here to measure with. So the first work wasn't retrieval at all; it was building something honest to measure against.

## 2. How each change was judged

The whole project runs on one rule: **capture a baseline, change one thing, re-run the same eval, record the delta.** A few decisions make that trustworthy.

**Retrieval and generation are measured separately.** Most RAG failures are retrieval failures wearing a generation costume — the model answers fluently from the wrong context. So the harness scores them apart: ranking metrics (precision/recall/MRR/nDCG) over retrieved documents, and a fact-by-fact judge over the generated answer. Ranking metrics are averaged only over queries that *have* a gold document; the negative "no-interaction" queries are graded on answer safety instead.

**The judge is calibrated, not assumed.** An unchecked LLM judge is a second opinion of unknown quality. A human-labeled subset is scored by the judge and the harness reports **judge-vs-human agreement %**. The first judge tried, local `llama3.1`, agreed only **68.6%** of the time — not trustworthy — so it was swapped for local `qwen2.5:7b`, which reached **80.4%**. That swap is the whole point of calibration: the number that appears in the results table comes from an instrument that was itself validated first, and it stayed free and local.

**Everything is reproducible and offline-testable.** Each stage sits behind a `Protocol` with a fake, and heavy dependencies (Pinecone, Ollama, sentence-transformers, Cohere) are lazy-loaded, so the full unit/API suite runs with no keys and no models — **112 ML + 5 Express + 6 frontend** tests, green in CI on every push. The work was done test-first, one reviewed change at a time.

## 3. Results

Each experiment rebuilds its own index namespace and is scored self-contained, so read the comparisons **within** each block. The keyword baseline is the **reset** baseline — captured after the corpus and eval set were expanded (§4.1), so it is deliberately harder than an eight-document toy.

### Retrieval: keyword → dense

Judge-independent, so these are the most solid numbers.

| metric | keyword baseline | BGE dense | delta |
|---|---|---|---|
| recall@5 | 0.5676 | **0.8919** | +0.3243 |
| precision@5 | 0.1586 | **0.3455** | +0.1869 |
| nDCG@5 | 0.4822 | **0.8265** | +0.3443 |
| MRR | 0.4536 | **0.8050** | +0.3514 |
| retrieval coverage | 0.2838 | **0.5946** | +0.3108 |

### Generation: judge, once calibrated

| metric | keyword baseline | BGE dense (Ollama + calibrated judge) |
|---|---|---|
| answer fact-coverage | 0.00 (no LLM) | **0.5674** |
| `must_not_say` violations | 1 | **5** |
| judge-vs-human agreement | — | **80.4%** (`qwen2.5:7b`) |

### Chunking (3-way ablation)

| strategy | nDCG@5 | recall@5 | precision@5 |
|---|---|---|---|
| fixed-512 | 0.7373 | 0.8108 | 0.3095 |
| **recursive** | **0.8555** | **0.9459** | **0.3824** |
| semantic | 0.7474 | 0.8108 | 0.2252 |

Recursive wins clearly; semantic over-merged (58 chunks vs 135) and its precision collapsed. **Caveat (see §5):** the `fixed` row here (0.7373) is suspiciously below the identical-chunker dense run above (0.8265), so the *magnitude* of recursive's lead is likely inflated by a Pinecone indexing-latency artifact — recursive is the winner, but treat "+0.12 nDCG" as an upper bound.

### Hybrid + rerank (alpha sweep, on fixed chunks)

| config | nDCG@5 | recall@5 | precision@5 |
|---|---|---|---|
| hybrid α=0.0 (pure BM25) | 0.5597 | 0.6486 | 0.2527 |
| hybrid α=0.5 | 0.8312 | 0.9189 | 0.3626 |
| **hybrid α=0.75 (winner)** | **0.8530** | **0.9189** | **0.3554** |
| hybrid α=1.0 (pure dense) | 0.8265 | 0.8919 | 0.3455 |
| hybrid α=0.75 **+ rerank** | 0.6337 | 0.7568 | 0.2599 |

`α=1.0` reproduces the dense run exactly (0.8265) — a clean sanity check that the hybrid path is consistent. The best blend, **α=0.75**, beats pure dense by **+0.027 nDCG** with better MRR: a modest but real hybrid gain. The reranker is the headline negative result below.

## 4. What changed, and what it bought

### 4.1 A data foundation worth measuring against

The eight-document corpus and thirteen-query set the prototype left behind couldn't tell a good retriever from a lucky one. So before any retrieval work, the corpus grew to **27 cited interaction monographs** (original prose, one file per interaction, with severity, mechanism, management, and citations) and the eval set to **47 labeled queries** — written to exercise the failure modes that matter here: brand-vs-generic names, dosage qualifiers, and **negative pairs with no real interaction**, so precision has something to be wrong about. A ~15-query human-labeled slice was added for judge calibration.

Re-running the keyword baseline on this harder set dropped recall from the toy corpus's inflated ~0.92 to **0.5676** and precision to **0.1586**. That drop is the point: the eval finally *discriminates*, so every lift after it is measured against a line that can actually be beaten honestly.

### 4.2 Dense retrieval and a judge you can trust

With a corpus to retrieve over, the pipeline moved to **BGE (`bge-large-en-v1.5`, 1024-d) on Pinecone**, per-chunk metadata, and namespaces. Generation moved to **local Ollama**, and the previously-stubbed judge became real.

The measured result is the cleanest win in the project: **recall@5 0.57 → 0.89, precision@5 0.16 → 0.35, nDCG@5 0.48 → 0.83**, and answer fact-coverage from a structural 0.00 to **0.57**. Dense retrieval understands that "blood thinner" and "anticoagulant" are the same idea in a way keyword overlap never will. And calibration earned its keep: the first local judge sat at 68.6% agreement, so its fact-coverage number couldn't be trusted; swapping to `qwen2.5:7b` (80.4%) produced a number that can be. One honest blemish: **5 of 47 answers still contained a `must_not_say` phrase** — a real generation-safety signal, and part of the motivation for the retrieval work that follows.

### 4.3 Chunking, decided by data instead of by taste

Three strategies were run head to head — fixed-size, LangChain recursive, semantic-breakpoint — each re-indexed and scored on the same eval. **Recursive won** (nDCG 0.8555, recall 0.9459), and semantic's aggressive merging (58 chunks) tanked its precision to 0.2252. The lesson holds even with the §5 caveat: chunking is not a set-and-forget default, and the right choice is an empirical question the harness answers in one run.

### 4.4 Hybrid retrieval, and a reranker that didn't earn its place

Dense embeddings are strong on meaning and weak on exact tokens (drug names, dosages, "INR"); BM25 is the mirror image. So the pipeline added BM25 sparse vectors and Pinecone hybrid, and rather than guess the blend, **swept it**: the best weight was **α=0.75**, a small real gain over pure dense (+0.027 nDCG).

Then a two-stage rerank was layered on: retrieve wide, rescore with a cross-encoder (`bge-reranker-base`), keep the top five. **It made things worse** — nDCG 0.853 → 0.634, recall 0.919 → 0.757. The small reranker was demoting correct documents out of the top-5, crowding it with textually-similar chunks from the wrong monographs. The right move was to **not adopt it** and say so. Reranking is not free lunch: it only helps when it's more discriminating than the retrieval underneath it, and here the base retrieval was already good and the reranker was underpowered. A stronger cross-encoder (e.g. `bge-reranker-v2-m3`, or Cohere Rerank — both pluggable via one env var) is the obvious next thing to test; the harness would settle it in one run.

### 4.5 Streaming, end to end

The legacy FAISS service was retired and replaced with a thin FastAPI app (`/query`, `/query/stream`); the old interaction-check endpoint was re-pointed onto the RAG pipeline while preserving the exact response contract the dashboard consumed. Answers stream token-by-token over SSE, through an Express passthrough, into a new "Ask ClearRx" panel that renders text as it arrives with its citations. A test enforces the invariant that the **streamed answer is byte-identical to the non-streamed one**, so streaming is purely a UX win and never a correctness risk; citations come from the retrieved chunks, not the model's prose, so they're grounded even if generation drifts.

## 5. Honest limitations

- **The reranker hurt** (§4.4). Reported as a negative result and dropped, not buried.
- **Pinecone indexing latency skewed one cross-experiment comparison.** The chunking run's `fixed` baseline (nDCG 0.7373) is well below the identical-chunker dense run (0.8265), because each experiment builds a namespace and queries it immediately, and Pinecone serverless has a short upsert→queryable delay — so the first strategy evaluated reads an under-indexed namespace. The *within-experiment* winners (recursive; hybrid α=0.75) still hold, but the absolute magnitudes across experiments aren't directly comparable. A clean re-run would build all namespaces, pause, then evaluate. This is exactly the kind of thing the separate-signals, committed-reports discipline makes visible instead of hiding.
- **The judge clears the bar, barely** (80.4%). Fact-coverage is trustworthy enough to quote but not to over-interpret; a stronger judge would tighten it.
- **The corpus is curated original prose** (~27 interactions), not an exhaustive clinical database. This is an engineering/eval exercise; the medical-advice disclaimer stays and no new clinical claims are made.
- **The live `/query` path serves dense retrieval.** Promoting the hybrid winner into the answer path is a deliberate, documented manual step (it needs the BM25 params loaded and a dump→load round-trip validated first), not silently switched on.

## 6. What this demonstrates

Less a retrieval trick, more a way of working: build the measurement before the feature; separate the signals so a number points at a cause; validate the instrument doing the measuring; let data pick chunking strategy and hybrid weight; and when a change (the reranker) or an artifact (indexing latency) makes results worse or noisier, **report it** instead of dressing it up. The metrics table is the artifact; the harness that produced it — and the honesty about what it revealed — is the actual deliverable.

## 7. Reproduce

Offline, no keys: `cd ml && source venv/bin/activate && python -m pytest -q` (this is what CI runs).
Live metrics: follow [`../docs/live-runs-provisioning.md`](../docs/live-runs-provisioning.md); the `eval/reports/*.{json,md}` it generates are the source of every number above.
