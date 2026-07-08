# ClearRx Phase 2 — Chunking Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two more chunking strategies (recursive, semantic) alongside the existing fixed-size chunker, and build a retrieval-only experiment harness that indexes each strategy into its own Pinecone namespace, evaluates retrieval, and reports which strategy wins — chosen empirically at run time.

**Architecture:** Three `Chunker` implementations behind the existing `Chunker` Protocol. Fixed stays per-section (the current baseline); Recursive and Semantic operate on the whole concatenated monograph so they choose their own boundaries. A `run_chunking_experiment.py` script re-indexes the corpus once per strategy (namespace `chunk_fixed` / `chunk_recursive` / `chunk_semantic`) and runs a **retrieval-only** eval (no generation/judge — chunking is a retrieval-stage change, so this isolates the signal, needs no Ollama, and is cheap). The winning strategy is picked by nDCG@10 (tie-break recall@5) at run time and documented; it is not hardcoded.

**Tech Stack:** Python 3.9.6, pydantic v2, pytest, sentence-transformers (BGE, from Phase 1b), Pinecone (from Phase 1b), LangChain `RecursiveCharacterTextSplitter` (`langchain-text-splitters`), numpy.

## Global Constraints

- Python **3.9.6** — every module starts with `from __future__ import annotations`.
- **pydantic v2** for all models.
- **Heavy deps stay lazy + injectable:** LangChain is imported *inside* the recursive chunker (never module-top) and the chunker accepts an injected `splitter=`; the semantic chunker accepts an injected `embedder`. Unit tests use fakes and never import langchain, sentence-transformers, or pinecone.
- **Unit tests** (`tests/unit`) are offline/fakes-only and run anywhere with no installs. The **experiment run** needs BGE + a live Pinecone index and runs **only on the provisioned machine** (after Phase 1b's index exists).
- Reuse existing modules — do NOT duplicate: `app/rag/chunking.py` (`Chunker`, `FixedSizeChunker`, `chunk_metadata`), `app/rag/models.py` (`Monograph`, `Chunk`), `app/rag/embeddings.py` (`Embedder`), `app/rag/vectorstore.py` (`PineconeStore`, `Record`), `app/rag/pipeline.py` (`DenseRagPipeline`), `app/ingest/build_index.py` (`build_records`), `app/eval/dataset.py` (`load_queries`), `app/eval/metrics.py` (`retrieval_coverage`, `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`).
- Whole-monograph chunkers set `section="document"` on their chunks (chunks may span sections; `source_doc_id` and the doc-level metadata are what retrieval uses, so section granularity is not needed there).
- Work runs from `ml/` with the venv active: `cd ml && source venv/bin/activate`. Run tests with `python -m pytest`. Commit after every task.

## Prerequisites (on the provisioned machine, for the experiment run only)

Phase 1b must already be run (deps installed, `PINECONE_API_KEY` set, `python -m app.ingest.build_index` done). Then `pip install -r requirements.txt` again to pull `langchain-text-splitters` (added in Task 1).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `ml/requirements.txt` | deps | Modify: add `langchain-text-splitters` |
| `ml/app/config.py` | settings | Modify: recursive size/overlap, semantic percentile |
| `ml/app/rag/chunking.py` | chunkers | Modify: add `RecursiveChunker`, `SemanticChunker`, `build_chunkers` |
| `ml/scripts/run_chunking_experiment.py` | experiment harness | Create |
| `ml/tests/unit/test_chunking.py` | chunker unit tests | Modify: add recursive/semantic/factory tests |
| `ml/tests/unit/test_chunking_experiment.py` | harness unit tests | Create |
| `ml/tests/unit/test_config.py` | config defaults | Modify: assert new fields |

---

### Task 1: Config fields + LangChain dependency

**Files:**
- Modify: `ml/requirements.txt`
- Modify: `ml/app/config.py`
- Test: `ml/tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` gains `chunk_recursive_size: int = 300`, `chunk_recursive_overlap: int = 60`, `semantic_threshold_percentile: float = 85.0`.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_config.py`:

```python
def test_phase2_chunking_settings_defaults():
    from app.config import Settings
    s = Settings()
    assert s.chunk_recursive_size == 300
    assert s.chunk_recursive_overlap == 60
    assert s.semantic_threshold_percentile == 85.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_config.py::test_phase2_chunking_settings_defaults -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'chunk_recursive_size'`

- [ ] **Step 3: Implement**

In `ml/requirements.txt`, add one line:

```
langchain-text-splitters>=0.2.0
```

In `ml/app/config.py`, add to `Settings` (after the existing `openai_api_key` line from Phase 1b):

```python
    chunk_recursive_size: int = 300
    chunk_recursive_overlap: int = 60
    semantic_threshold_percentile: float = 85.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt app/config.py tests/unit/test_config.py
git commit -m "feat(ml): add Phase 2 chunking config + langchain-text-splitters dep"
```

---

### Task 2: `RecursiveChunker`

Wrap LangChain's `RecursiveCharacterTextSplitter`. Operate on the whole monograph (sections concatenated with their names as lightweight headers), lazy-import LangChain, and accept an injected `splitter` so unit tests use a fake.

**Files:**
- Modify: `ml/app/rag/chunking.py`
- Test: `ml/tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `chunk_metadata(doc)`, `Chunk`, `Monograph` (already in the module).
- Produces: `class RecursiveChunker` with `name = "recursive"`, `__init__(self, chunk_size: int = 300, overlap: int = 60, *, splitter=None)`, `chunk(self, doc: Monograph) -> list[Chunk]`. The injected `splitter` (or the lazy LangChain one) exposes `split_text(text: str) -> list[str]`. Chunks carry `section="document"`.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_chunking.py`:

```python
from app.rag.chunking import RecursiveChunker
from app.rag.models import Monograph


class FakeSplitter:
    """Stands in for LangChain's RecursiveCharacterTextSplitter."""
    def __init__(self):
        self.seen = None
    def split_text(self, text):
        self.seen = text
        return ["piece one", "piece two"]


def _doc():
    return Monograph(
        id="int_a_b", drug_a="a", drug_b="b", drug_class_a="x", drug_class_b="y",
        severity="high",
        sections={"summary": "a and b interact", "management": "avoid combining"},
    )


def test_recursive_chunker_uses_splitter_over_whole_document():
    fake = FakeSplitter()
    chunker = RecursiveChunker(splitter=fake)
    chunks = chunker.chunk(_doc())
    assert chunker.name == "recursive"
    assert [c.text for c in chunks] == ["piece one", "piece two"]
    # whole-document input: both section texts are present in what the splitter saw
    assert "a and b interact" in fake.seen and "avoid combining" in fake.seen
    assert [c.chunk_index for c in chunks] == [0, 1]
    for c in chunks:
        assert c.source_doc_id == "int_a_b"
        assert c.section == "document"
        assert c.metadata["severity"] == "high"
        assert "a" in c.metadata["drugs_mentioned"]


def test_recursive_chunker_drops_empty_pieces():
    class Emptyish:
        def split_text(self, text):
            return ["real chunk", "   ", ""]
    chunks = RecursiveChunker(splitter=Emptyish()).chunk(_doc())
    assert [c.text for c in chunks] == ["real chunk"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_chunking.py::test_recursive_chunker_uses_splitter_over_whole_document -v`
Expected: FAIL — `ImportError: cannot import name 'RecursiveChunker'`

- [ ] **Step 3: Implement**

Append to `ml/app/rag/chunking.py`:

```python
class RecursiveChunker:
    """Recursive character splitting over the whole monograph using
    LangChain's RecursiveCharacterTextSplitter. Sections are concatenated
    with their names as light headers so the splitter can choose boundaries
    across the document rather than being confined to one section."""

    name = "recursive"

    def __init__(self, chunk_size: int = 300, overlap: int = 60, *, splitter=None) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = splitter

    def _get_splitter(self):
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # lazy

            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        return self._splitter

    def chunk(self, doc: Monograph) -> list[Chunk]:
        md = chunk_metadata(doc)
        text = "\n\n".join(
            f"{section}. {body}" for section, body in doc.sections.items() if body.strip()
        )
        pieces = [p for p in self._get_splitter().split_text(text) if p.strip()]
        return [
            Chunk(
                text=piece,
                source_doc_id=doc.id,
                section="document",
                chunk_index=i,
                metadata=dict(md),
            )
            for i, piece in enumerate(pieces)
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_chunking.py -v`
Expected: PASS (existing fixed-chunker tests + the two new recursive tests)

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunking.py tests/unit/test_chunking.py
git commit -m "feat(ml): add recursive (LangChain) whole-document chunker"
```

---

### Task 3: `SemanticChunker`

Split the whole monograph into sentences, embed each, and break where the cosine distance between consecutive sentences exceeds the configured percentile of all consecutive distances. Accept an injected `embedder` so unit tests use a fake and never load BGE.

**Files:**
- Modify: `ml/app/rag/chunking.py`
- Test: `ml/tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `chunk_metadata`, `Chunk`, `Monograph`, and an `Embedder` (from `app/rag/embeddings.py`) whose `embed(texts) -> np.ndarray` returns L2-normalized rows.
- Produces: `class SemanticChunker` with `name = "semantic"`, `__init__(self, embedder, threshold_percentile: float = 85.0)`, `chunk(self, doc: Monograph) -> list[Chunk]`. Chunks carry `section="document"`.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_chunking.py`:

```python
import numpy as np
from app.rag.chunking import SemanticChunker


class ClusterEmbedder:
    """Maps sentences to one of two orthogonal unit vectors based on a keyword,
    so a semantic boundary falls exactly between the two groups."""
    dimension = 2

    def embed(self, texts):
        rows = []
        for t in texts:
            rows.append([1.0, 0.0] if "alpha" in t else [0.0, 1.0])
        return np.array(rows, dtype=np.float32)

    def embed_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def _two_topic_doc():
    return Monograph(
        id="int_a_b", drug_a="a", drug_b="b", drug_class_a="x", drug_class_b="y",
        severity="moderate",
        sections={
            "summary": "First alpha sentence here. Second alpha sentence here.",
            "mechanism": "First beta sentence here. Second beta sentence here.",
        },
    )


def test_semantic_chunker_breaks_at_topic_boundary():
    chunker = SemanticChunker(ClusterEmbedder(), threshold_percentile=85.0)
    chunks = chunker.chunk(_two_topic_doc())
    assert chunker.name == "semantic"
    # the two alpha sentences group together; the two beta sentences group together
    assert len(chunks) == 2
    assert "alpha" in chunks[0].text and "beta" not in chunks[0].text
    assert "beta" in chunks[1].text and "alpha" not in chunks[1].text
    for c in chunks:
        assert c.source_doc_id == "int_a_b"
        assert c.section == "document"
        assert c.metadata["severity"] == "moderate"


def test_semantic_chunker_single_sentence_is_one_chunk():
    doc = Monograph(
        id="int_c_d", drug_a="c", drug_b="d", drug_class_a="x", drug_class_b="y",
        severity="low", sections={"summary": "Only one sentence"},
    )
    chunks = SemanticChunker(ClusterEmbedder()).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_chunking.py::test_semantic_chunker_breaks_at_topic_boundary -v`
Expected: FAIL — `ImportError: cannot import name 'SemanticChunker'`

- [ ] **Step 3: Implement**

Add these imports at the top of `ml/app/rag/chunking.py` if not already present (the module already has `from __future__ import annotations` and imports `Chunk`, `Monograph`):

```python
import re

import numpy as np
```

Then append:

```python
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """Semantic-breakpoint chunking over the whole monograph. Sentences are
    embedded; a boundary is placed wherever the cosine distance between
    consecutive sentences reaches the configured percentile of all such
    distances. Embeddings are assumed L2-normalized, so cosine distance is
    1 - dot product."""

    name = "semantic"

    def __init__(self, embedder, threshold_percentile: float = 85.0) -> None:
        self.embedder = embedder
        self.threshold_percentile = threshold_percentile

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

    def chunk(self, doc: Monograph) -> list[Chunk]:
        md = chunk_metadata(doc)
        text = " ".join(body for body in doc.sections.values() if body.strip())
        sentences = self._sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [Chunk(text=sentences[0], source_doc_id=doc.id, section="document",
                          chunk_index=0, metadata=dict(md))]

        vecs = self.embedder.embed(sentences)
        distances = [
            1.0 - float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(sentences) - 1)
        ]
        threshold = float(np.percentile(distances, self.threshold_percentile))

        groups: list[list[str]] = [[sentences[0]]]
        for i, dist in enumerate(distances):
            if threshold > 0 and dist >= threshold:
                groups.append([sentences[i + 1]])
            else:
                groups[-1].append(sentences[i + 1])

        texts = [" ".join(g).strip() for g in groups]
        return [
            Chunk(text=t, source_doc_id=doc.id, section="document", chunk_index=i,
                  metadata=dict(md))
            for i, t in enumerate(texts)
            if t
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_chunking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunking.py tests/unit/test_chunking.py
git commit -m "feat(ml): add semantic-breakpoint whole-document chunker"
```

---

### Task 4: `build_chunkers` factory

Provide the ordered list of strategies the experiment iterates over.

**Files:**
- Modify: `ml/app/rag/chunking.py`
- Test: `ml/tests/unit/test_chunking.py`

**Interfaces:**
- Consumes: `Settings` (`chunk_recursive_size`, `chunk_recursive_overlap`, `semantic_threshold_percentile`) and an `Embedder`.
- Produces: `def build_chunkers(settings, embedder) -> list` returning `[FixedSizeChunker(chunk_size=512, overlap=0), RecursiveChunker(...), SemanticChunker(...)]` with `.name` values `"fixed"`, `"recursive"`, `"semantic"`. The fixed config (512/0) matches Phase 1b's `build_index` baseline.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_chunking.py`:

```python
from app.config import Settings
from app.rag.chunking import build_chunkers


def test_build_chunkers_returns_three_named_strategies():
    chunkers = build_chunkers(Settings(), ClusterEmbedder())
    assert [c.name for c in chunkers] == ["fixed", "recursive", "semantic"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_chunking.py::test_build_chunkers_returns_three_named_strategies -v`
Expected: FAIL — `ImportError: cannot import name 'build_chunkers'`

- [ ] **Step 3: Implement**

Append to `ml/app/rag/chunking.py`:

```python
def build_chunkers(settings, embedder) -> list:
    return [
        FixedSizeChunker(chunk_size=512, overlap=0),
        RecursiveChunker(settings.chunk_recursive_size, settings.chunk_recursive_overlap),
        SemanticChunker(embedder, settings.semantic_threshold_percentile),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_chunking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/chunking.py tests/unit/test_chunking.py
git commit -m "feat(ml): add chunker factory (fixed/recursive/semantic)"
```

---

### Task 5: Chunking experiment harness

A script that, per strategy, re-indexes the corpus into a dedicated namespace, runs a **retrieval-only** eval, and reports a comparison table plus the winner. The pure pieces (`evaluate_retrieval`, `compare_strategies`, `pick_winner`) are unit-tested with fakes; `main()` is the live run (deferred to the provisioned machine).

**Files:**
- Create: `ml/scripts/run_chunking_experiment.py`
- Test: `ml/tests/unit/test_chunking_experiment.py`

**Interfaces:**
- Consumes: `build_chunkers`, `build_records` (`app/ingest/build_index.py`), `PineconeStore`, `BGEEmbedder`, `DenseRagPipeline`, `load_queries`, `EvalQuery`, and the metric functions in `app/eval/metrics.py`.
- Produces:
  - `def evaluate_retrieval(retriever, queries: list, k: int) -> dict` — returns aggregate `{"retrieval_coverage","precision_at_k","recall_at_k","mrr","ndcg","n_queries","n_retrieval_gradable"}`, ranking metrics averaged only over queries with non-empty `expected_doc_ids`, coverage over queries with non-empty `expected_retrieval_topics` (mirrors the Phase 1b runner's gating). `retriever` is any object with `.retrieve(query: str, k: int) -> list[Chunk]`.
  - `def compare_strategies(results: dict) -> list[dict]` — `results` maps strategy name → aggregate dict; returns one row per strategy: `{"strategy", "retrieval_coverage", "precision_at_k", "recall_at_k", "mrr", "ndcg"}`.
  - `def pick_winner(results: dict, metric: str = "ndcg", tiebreak: str = "recall_at_k") -> str` — the strategy name maximizing `metric`, ties broken by `tiebreak`.
  - `def main() -> None`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_chunking_experiment.py`:

```python
from __future__ import annotations

from app.eval.dataset import EvalQuery
from app.rag.models import Chunk
from scripts.run_chunking_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner,
)


class OneDocRetriever:
    """Always returns chunks from the given doc id, so retrieval metrics are
    fully determined by whether that doc is the gold doc."""
    def __init__(self, doc_id):
        self.doc_id = doc_id
    def retrieve(self, query, k):
        return [Chunk(text="nsaid anticoagulant bleeding", source_doc_id=self.doc_id,
                      section="document", chunk_index=0)]


def _queries():
    return [
        EvalQuery(id="q1", query="warfarin ibuprofen", query_type="interaction",
                  expected_doc_ids=["int_warfarin_ibuprofen"],
                  expected_retrieval_topics=["nsaid anticoagulant"],
                  expected_answer_facts=["bleeding"], must_not_say=[], severity="high"),
        EvalQuery(id="n1", query="amoxicillin acetaminophen", query_type="no_interaction",
                  expected_doc_ids=[], expected_retrieval_topics=[],
                  expected_answer_facts=["no interaction"], must_not_say=[], severity="low"),
    ]


def test_evaluate_retrieval_gates_on_gradability():
    agg = evaluate_retrieval(OneDocRetriever("int_warfarin_ibuprofen"), _queries(), k=5)
    assert agg["n_queries"] == 2
    assert agg["n_retrieval_gradable"] == 1        # only the positive query has a gold doc
    assert agg["recall_at_k"] == 1.0               # averaged over the 1 gradable query
    assert agg["mrr"] == 1.0
    assert agg["retrieval_coverage"] == 1.0        # "nsaid anticoagulant" is in the chunk text


def test_pick_winner_by_ndcg_with_recall_tiebreak():
    results = {
        "fixed": {"ndcg": 0.60, "recall_at_k": 0.50},
        "recursive": {"ndcg": 0.72, "recall_at_k": 0.66},
        "semantic": {"ndcg": 0.72, "recall_at_k": 0.61},
    }
    assert pick_winner(results) == "recursive"     # ties on ndcg (0.72) broken by recall


def test_compare_strategies_emits_one_row_per_strategy():
    results = {
        "fixed": {"retrieval_coverage": 0.5, "precision_at_k": 0.2, "recall_at_k": 0.5,
                  "mrr": 0.4, "ndcg": 0.45},
        "recursive": {"retrieval_coverage": 0.6, "precision_at_k": 0.25, "recall_at_k": 0.66,
                      "mrr": 0.5, "ndcg": 0.55},
    }
    rows = compare_strategies(results)
    assert {r["strategy"] for r in rows} == {"fixed", "recursive"}
    fixed = next(r for r in rows if r["strategy"] == "fixed")
    assert fixed["ndcg"] == 0.45
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_chunking_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_chunking_experiment'`

- [ ] **Step 3: Implement**

Create `ml/scripts/run_chunking_experiment.py`:

```python
from __future__ import annotations

import json
import os

from app.config import get_settings
from app.eval.dataset import load_queries
from app.eval.metrics import (
    ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, retrieval_coverage,
)
from app.ingest.build_index import build_records
from app.rag.chunking import build_chunkers
from app.rag.embeddings import BGEEmbedder
from app.rag.pipeline import DenseRagPipeline
from app.rag.vectorstore import PineconeStore

_METRICS = ["retrieval_coverage", "precision_at_k", "recall_at_k", "mrr", "ndcg"]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _distinct_doc_ids(chunks) -> list[str]:
    ordered: list[str] = []
    seen: set = set()
    for c in chunks:
        if c.source_doc_id not in seen:
            seen.add(c.source_doc_id)
            ordered.append(c.source_doc_id)
    return ordered


def evaluate_retrieval(retriever, queries: list, k: int) -> dict:
    cov, prec, rec, mrr, ndcg = [], [], [], [], []
    gradable = 0
    for q in queries:
        chunks = retriever.retrieve(q.query, k)
        ids = _distinct_doc_ids(chunks)
        relevant = set(q.expected_doc_ids)
        if q.expected_retrieval_topics:
            cov.append(retrieval_coverage(q.expected_retrieval_topics, [c.text for c in chunks]))
        if relevant:
            gradable += 1
            prec.append(precision_at_k(ids, relevant, k))
            rec.append(recall_at_k(ids, relevant, k))
            mrr.append(reciprocal_rank(ids, relevant))
            ndcg.append(ndcg_at_k(ids, relevant, k))
    return {
        "retrieval_coverage": _mean(cov),
        "precision_at_k": _mean(prec),
        "recall_at_k": _mean(rec),
        "mrr": _mean(mrr),
        "ndcg": _mean(ndcg),
        "n_queries": float(len(queries)),
        "n_retrieval_gradable": float(gradable),
    }


def compare_strategies(results: dict) -> list[dict]:
    rows = []
    for name, agg in results.items():
        row = {"strategy": name}
        for m in _METRICS:
            row[m] = agg[m]
        rows.append(row)
    return rows


def pick_winner(results: dict, metric: str = "ndcg", tiebreak: str = "recall_at_k") -> str:
    return max(results, key=lambda name: (results[name][metric], results[name][tiebreak]))


def _to_markdown(rows: list[dict], winner: str) -> str:
    header = "| strategy | " + " | ".join(_METRICS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in _METRICS) + " |"
    lines = [header, sep]
    for r in rows:
        cells = " | ".join(f"{r[m]:.4f}" for m in _METRICS)
        lines.append(f"| {r['strategy']} | {cells} |")
    lines.append("")
    lines.append(f"**Winner (nDCG, recall tie-break): {winner}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    settings = get_settings()
    queries = load_queries(os.path.join(settings.eval_dir, "queries.json"))
    from app.rag.corpus import load_corpus

    docs = load_corpus(settings.corpus_dir)
    embedder = BGEEmbedder(settings.embedding_model, settings.embedding_dim)
    store = PineconeStore(settings.pinecone_api_key, settings.pinecone_index)
    store.ensure_index(settings.embedding_dim, settings.pinecone_metric,
                       settings.pinecone_cloud, settings.pinecone_region)

    results: dict = {}
    for chunker in build_chunkers(settings, embedder):
        namespace = f"chunk_{chunker.name}"
        records = build_records(docs, chunker, embedder)
        store.upsert(records, namespace=namespace)
        pipeline = DenseRagPipeline(embedder, store, llm=None, namespace=namespace)
        results[chunker.name] = evaluate_retrieval(pipeline, queries, k=5)
        print(f"{chunker.name}: {len(records)} chunks -> {results[chunker.name]}")

    rows = compare_strategies(results)
    winner = pick_winner(results)
    os.makedirs(settings.reports_dir, exist_ok=True)
    with open(os.path.join(settings.reports_dir, "chunking.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": results, "winner": winner}, fh, indent=2)
    md = _to_markdown(rows, winner)
    with open(os.path.join(settings.reports_dir, "chunking.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
```

Note: `DenseRagPipeline` is constructed with `llm=None` because the experiment only calls `.retrieve` (chunking is a retrieval-stage change); `.generate` is never invoked, so no LLM is needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_chunking_experiment.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Run the full offline suite**

Run: `python -m pytest tests/unit tests/eval -q`
Expected: PASS (all offline tests green; no new integration tests added this phase).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_chunking_experiment.py tests/unit/test_chunking_experiment.py
git commit -m "feat(ml): add retrieval-only chunking experiment harness"
```

---

## Operator run book (on the provisioned machine, after all 5 tasks)

Phase 1b must already have been run once (index created, deps installed).

```bash
cd ml && source venv/bin/activate
git pull
pip install -r requirements.txt          # pulls langchain-text-splitters
python -m pytest tests/unit tests/eval -q # offline gate

# run the experiment: re-indexes each strategy into its own namespace, retrieval-evals, compares
python -m scripts.run_chunking_experiment
```

This writes `eval/reports/chunking.{json,md}` — a per-strategy retrieval-metrics table and the empirically-chosen winner (highest nDCG@10, recall@5 tie-break). The `curated` namespace used by the main pipeline is untouched (strategies live in `chunk_fixed`/`chunk_recursive`/`chunk_semantic`).

**Adopting the winner (manual follow-up):** once the winner is known, set it as the default chunker in `app/ingest/build_index.py` (replace `FixedSizeChunker(chunk_size=512, overlap=0)` with the winning chunker — for recursive use `RecursiveChunker(settings.chunk_recursive_size, settings.chunk_recursive_overlap)`; for semantic use `SemanticChunker(embedder, settings.semantic_threshold_percentile)`) and re-run `python -m app.ingest.build_index` to rebuild the `curated` namespace. Record the delta in the README debrief (Phase 5).

**If the result is flat** (all strategies within noise — plausible given the corpus's sections are already short semantic units), document that honestly rather than manufacturing a winner: keep fixed-size for simplicity and note the finding. This is an acceptable, spec-anticipated outcome (design spec §11).

---

## Self-Review

- **Spec coverage (design §8 Step 3):** three chunkers — fixed (existing), recursive (Task 2), semantic (Task 3); factory (Task 4); re-index + eval each and compare with the coverage/precision/recall/mrr/nDCG delta (Task 5); empirical winner selection + honest-flat-result handling (run book). LangChain `RecursiveCharacterTextSplitter` used per the build doc's Step 3.
- **Deferred / out of scope:** hybrid + rerank (Phase 3), streaming (Phase 4), README debrief (Phase 5). Generation/judge are intentionally excluded from the experiment (retrieval-only), which the plan states and justifies.
- **Placeholder scan:** none — every module and its primary unit test carry complete, runnable code.
- **Type consistency:** `Chunker.name`/`chunk(doc)`, `RecursiveChunker(chunk_size, overlap, *, splitter)`, `SemanticChunker(embedder, threshold_percentile)`, `build_chunkers(settings, embedder)`, and the harness's `evaluate_retrieval(retriever, queries, k)` / `compare_strategies(results)` / `pick_winner(results, metric, tiebreak)` are used identically across tasks. Chunk fields (`text`/`source_doc_id`/`section`/`chunk_index`/`metadata`) and the `section="document"` convention match `app/rag/models.py` and what `build_records` (Phase 1b) writes. The gating logic in `evaluate_retrieval` mirrors the Phase 1b runner (ranking metrics over `expected_doc_ids`-bearing queries, coverage over `expected_retrieval_topics`-bearing queries).
