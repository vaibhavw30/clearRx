# ClearRx Phase 3 — Hybrid Retrieval + Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BM25 sparse retrieval + Pinecone hybrid search and a two-stage rerank (local `bge-reranker` cross-encoder by default, Cohere via one env var), plus a retrieval-only experiment harness that compares dense vs alpha-swept hybrid vs hybrid+rerank and picks the winner empirically.

**Architecture:** New stages sit behind Protocols with fakes (like Phase 1b/2). Heavy deps (`pinecone-text`, `cohere`, sentence-transformers `CrossEncoder`) are lazy-imported inside methods so the offline unit suite needs none installed. Sparse is strictly additive to the vector store (`sparse_values`/`sparse` default `None` → dense behavior unchanged). Two small DRY extractions (shared harness helpers; `chunks_from_matches`) keep the new code duplication-free. The live experiment run is deferred to the provisioned machine.

**Tech Stack:** Python 3.9.6, pydantic v2, pytest, sentence-transformers (BGE + CrossEncoder), Pinecone (dense+sparse hybrid), pinecone-text (BM25), cohere (optional rerank).

## Global Constraints

- Python **3.9.6** — every module starts with `from __future__ import annotations`.
- **pydantic v2** for all models.
- **Heavy deps stay lazy + injectable:** `pinecone`, `pinecone_text`, `cohere`, and `sentence_transformers` are imported *inside* methods, never at module top; every stage takes an injected collaborator (`encoder=`, `model=`, `client=`, `index=`) so unit tests use fakes and never install those packages.
- **Unit tests** (`tests/unit`) are offline/fakes-only and pass with none of pinecone/pinecone-text/cohere/ollama installed. The **experiment run** needs BGE + a live Pinecone index + the reranker model, and runs **only on the provisioned machine**.
- **Sparse is additive/back-compatible:** `Record.sparse_values` and `VectorStore.query(..., sparse=)` both default `None`; the dense-only code path (`DenseRagPipeline`, Phase 2 harness) must stay byte-for-byte behaviorally unchanged, and their existing tests must stay green untouched.
- **Reuse, do not duplicate:** `app/eval/aggregate.py` (`mean`, `distinct_doc_ids`), `app/eval/metrics.py` (ranking metrics), `app/rag/models.py` (`Chunk`, `Monograph`), `app/rag/vectorstore.py` (`Record`, `Match`, `PineconeStore`), `app/rag/embeddings.py` (`BGEEmbedder`, `Embedder`), `app/ingest/build_index.py` (`build_records`), `app/eval/dataset.py` (`load_queries`).
- Work runs from `ml/` with the venv active: `cd ml && source venv/bin/activate`. Run tests with `python -m pytest`. Commit after every task.

## Prerequisites (experiment run only, provisioned machine)

Phase 1b already run (deps installed, `PINECONE_API_KEY` set). Then `pip install -r requirements.txt` to pull `pinecone-text` + `cohere` (added in Task 2), rebuild the `hybrid` namespace, and `python -m scripts.run_retrieval_experiment`. `bge-reranker-base` (~1GB) downloads on first `CrossEncoder` load; runs on CPU.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `ml/app/eval/retrieval_experiment.py` | shared retrieval-experiment harness helpers | Create (extracted from Phase 2 harness) |
| `ml/scripts/run_chunking_experiment.py` | Phase 2 harness | Modify: import shared helpers |
| `ml/app/config.py` | settings | Modify: rerank/hybrid/bm25 fields |
| `ml/requirements.txt` | deps | Modify: add `pinecone-text`, `cohere` |
| `ml/app/rag/sparse.py` | BM25 sparse encoder | Create |
| `ml/app/rag/rerank.py` | reranker Protocol + local/Cohere + factory | Create |
| `ml/app/rag/vectorstore.py` | vector store | Modify: sparse in `Record`/`query`/`upsert` |
| `ml/app/rag/retrieval.py` | `chunks_from_matches` helper | Create (extracted from pipeline) |
| `ml/app/rag/pipeline.py` | dense pipeline | Modify: use `chunks_from_matches` |
| `ml/app/rag/hybrid.py` | convex scale + hybrid+rerank retriever | Create |
| `ml/app/ingest/build_index.py` | index builder | Modify: `build_records` optional sparse |
| `ml/scripts/run_retrieval_experiment.py` | Phase 3 experiment harness | Create |
| (+ matching `tests/unit/test_*.py`) | unit tests | Create/Modify |

---

### Task 1: Extract shared retrieval-experiment helpers

The Phase 3 harness needs the same `evaluate_retrieval`/`compare_strategies`/`pick_winner`/markdown logic the Phase 2 harness already has. Promote them into a shared module both import (mirrors the Phase 2 `aggregate.py` extraction).

**Files:**
- Create: `ml/app/eval/retrieval_experiment.py`
- Modify: `ml/scripts/run_chunking_experiment.py`
- Test: `ml/tests/unit/test_retrieval_experiment.py` (create)

**Interfaces:**
- Produces in `app/eval/retrieval_experiment.py`: `_METRICS`, `evaluate_retrieval(retriever, queries, k) -> dict`, `compare_strategies(results: dict) -> list[dict]`, `pick_winner(results, metric="ndcg", tiebreak="recall_at_k") -> str`, `to_markdown(rows, winner) -> str`.
- `run_chunking_experiment.py` imports all of these from the new module and drops its local copies; its `main()` calls `to_markdown` (was `_to_markdown`). The existing `tests/unit/test_chunking_experiment.py` still imports `evaluate_retrieval`/`compare_strategies`/`pick_winner` from `scripts.run_chunking_experiment` — that keeps working because the script re-exports them via import, so **do not modify that test**.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_retrieval_experiment.py`:

```python
from __future__ import annotations

from app.eval.dataset import EvalQuery
from app.eval.retrieval_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner, to_markdown,
)
from app.rag.models import Chunk


class OneDocRetriever:
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
        EvalQuery(id="n1", query="amox acetaminophen", query_type="no_interaction",
                  expected_doc_ids=[], expected_retrieval_topics=[],
                  expected_answer_facts=["no interaction"], must_not_say=[], severity="low"),
    ]


def test_evaluate_retrieval_gates_on_gradability():
    agg = evaluate_retrieval(OneDocRetriever("int_warfarin_ibuprofen"), _queries(), k=5)
    assert agg["n_queries"] == 2
    assert agg["n_retrieval_gradable"] == 1
    assert agg["recall_at_k"] == 1.0
    assert agg["retrieval_coverage"] == 1.0


def test_pick_winner_ndcg_recall_tiebreak():
    results = {"a": {"ndcg": 0.6, "recall_at_k": 0.5},
               "b": {"ndcg": 0.72, "recall_at_k": 0.66},
               "c": {"ndcg": 0.72, "recall_at_k": 0.61}}
    assert pick_winner(results) == "b"


def test_compare_and_markdown():
    results = {"dense": {"retrieval_coverage": 0.5, "precision_at_k": 0.2,
                         "recall_at_k": 0.5, "mrr": 0.4, "ndcg": 0.45}}
    rows = compare_strategies(results)
    assert rows[0]["strategy"] == "dense" and rows[0]["ndcg"] == 0.45
    md = to_markdown(rows, "dense")
    assert "| strategy |" in md and "dense" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_retrieval_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.eval.retrieval_experiment'`

- [ ] **Step 3: Implement**

Create `ml/app/eval/retrieval_experiment.py` by **moving** the following definitions out of `ml/scripts/run_chunking_experiment.py` verbatim: `_METRICS`, `evaluate_retrieval`, `compare_strategies`, `pick_winner`, and the `_to_markdown` function **renamed to `to_markdown`**. The module needs these imports at top (copy from the script):

```python
from __future__ import annotations

from app.eval.aggregate import distinct_doc_ids, mean
from app.eval.metrics import (
    ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, retrieval_coverage,
)
```

Then in `ml/scripts/run_chunking_experiment.py`: delete those five definitions and their now-unused imports (`distinct_doc_ids, mean`, the `app.eval.metrics` group), add:

```python
from app.eval.retrieval_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner, to_markdown,
)
```

and change the one `_to_markdown(` call in its `main()` to `to_markdown(`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_retrieval_experiment.py tests/unit/test_chunking_experiment.py -v`
Expected: PASS — new module tests plus the unchanged Phase 2 harness tests (which still import the three helpers from `scripts.run_chunking_experiment`).

- [ ] **Step 5: Commit**

```bash
git add app/eval/retrieval_experiment.py scripts/run_chunking_experiment.py tests/unit/test_retrieval_experiment.py
git commit -m "refactor(ml): extract shared retrieval-experiment helpers"
```

---

### Task 2: Config fields + hybrid/rerank dependencies

**Files:**
- Modify: `ml/app/config.py`
- Modify: `ml/requirements.txt`
- Test: `ml/tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` gains `rerank_provider="local"`, `rerank_model_local="BAAI/bge-reranker-base"`, `cohere_api_key=""`, `cohere_rerank_model="rerank-english-v3.0"`, `hybrid_alphas=[0.0,0.25,0.5,0.75,1.0]`, `hybrid_top_k=50`, `hybrid_namespace="hybrid"`, `bm25_params_path` (under `data/`).

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_config.py`:

```python
def test_phase3_hybrid_rerank_settings_defaults():
    from app.config import Settings
    s = Settings()
    assert s.rerank_provider == "local"
    assert s.rerank_model_local == "BAAI/bge-reranker-base"
    assert s.cohere_rerank_model == "rerank-english-v3.0"
    assert s.hybrid_alphas == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert s.hybrid_top_k == 50
    assert s.hybrid_namespace == "hybrid"
    assert s.bm25_params_path.endswith("bm25_params.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_config.py::test_phase3_hybrid_rerank_settings_defaults -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'rerank_provider'`

- [ ] **Step 3: Implement**

In `ml/app/config.py`, add to `Settings` (after the `semantic_threshold_percentile` line):

```python
    rerank_provider: str = "local"
    rerank_model_local: str = "BAAI/bge-reranker-base"
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-english-v3.0"
    hybrid_alphas: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    hybrid_top_k: int = 50
    hybrid_namespace: str = "hybrid"
    bm25_params_path: str = os.path.normpath(os.path.join(_DATA, "bm25_params.json"))
```

In `ml/requirements.txt`, append two lines:

```
pinecone-text>=0.9.0
cohere>=5.0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py requirements.txt tests/unit/test_config.py
git commit -m "feat(ml): add Phase 3 hybrid/rerank config + pinecone-text, cohere deps"
```

---

### Task 3: `BM25SparseEncoder`

Wrap pinecone-text's `BM25Encoder`, lazy-imported, injectable for tests.

**Files:**
- Create: `ml/app/rag/sparse.py`
- Test: `ml/tests/unit/test_sparse.py`

**Interfaces:**
- Produces: `SparseEncoder` Protocol; `BM25SparseEncoder(*, encoder=None)` with `fit(corpus_texts: list[str]) -> None`, `encode_documents(texts: list[str]) -> list[dict]`, `encode_query(text: str) -> dict`, `dump(path: str) -> None`, `load(path: str) -> None`. Sparse dicts are Pinecone-format `{"indices": [...], "values": [...]}`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_sparse.py`:

```python
from __future__ import annotations

from app.rag.sparse import BM25SparseEncoder


class FakeBM25:
    def __init__(self):
        self.fitted = None
        self.dumped = None
        self.loaded = None
    def fit(self, corpus):
        self.fitted = list(corpus)
    def encode_documents(self, texts):
        return [{"indices": [1], "values": [0.5]} for _ in texts]
    def encode_query(self, text):
        return {"indices": [1], "values": [0.9]}
    def dump(self, path):
        self.dumped = path
    def load(self, path):
        self.loaded = path


def test_fit_and_encode_delegate_to_encoder():
    fake = FakeBM25()
    enc = BM25SparseEncoder(encoder=fake)
    enc.fit(["a b", "c d"])
    assert fake.fitted == ["a b", "c d"]
    docs = enc.encode_documents(["a b", "c d"])
    assert len(docs) == 2 and docs[0] == {"indices": [1], "values": [0.5]}
    assert enc.encode_query("a") == {"indices": [1], "values": [0.9]}


def test_dump_load_delegate(tmp_path):
    fake = FakeBM25()
    enc = BM25SparseEncoder(encoder=fake)
    p = str(tmp_path / "bm25.json")
    enc.dump(p)
    enc.load(p)
    assert fake.dumped == p and fake.loaded == p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_sparse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.sparse'`

- [ ] **Step 3: Implement**

Create `ml/app/rag/sparse.py`:

```python
from __future__ import annotations

from typing import Protocol


class SparseEncoder(Protocol):
    def fit(self, corpus_texts: list[str]) -> None: ...
    def encode_documents(self, texts: list[str]) -> list[dict]: ...
    def encode_query(self, text: str) -> dict: ...
    def dump(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


class BM25SparseEncoder:
    """BM25 sparse encoder backed by pinecone-text. Inject `encoder` in tests
    to avoid importing pinecone-text or fitting a real corpus."""

    def __init__(self, *, encoder=None) -> None:
        self._encoder = encoder

    def _enc(self):
        if self._encoder is None:
            from pinecone_text.sparse import BM25Encoder  # lazy

            self._encoder = BM25Encoder()
        return self._encoder

    def fit(self, corpus_texts: list[str]) -> None:
        self._enc().fit(corpus_texts)

    def encode_documents(self, texts: list[str]) -> list[dict]:
        return self._enc().encode_documents(texts)

    def encode_query(self, text: str) -> dict:
        return self._enc().encode_query(text)

    def dump(self, path: str) -> None:
        self._enc().dump(path)

    def load(self, path: str) -> None:
        self._enc().load(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_sparse.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/sparse.py tests/unit/test_sparse.py
git commit -m "feat(ml): add BM25 sparse encoder (pinecone-text, lazy)"
```

---

### Task 4: `Reranker` Protocol + local/Cohere impls + factory

Mirror the `build_judge` pattern (`app/rag/judge_clients.py`): a Protocol, two lazy-imported impls, and a factory that picks by settings with injectable overrides for tests.

**Files:**
- Create: `ml/app/rag/rerank.py`
- Test: `ml/tests/unit/test_rerank.py`

**Interfaces:**
- Produces: `Reranker` Protocol (`rerank(query: str, docs: list[str], top_n: int) -> list[int]` — indices into `docs`, best first); `LocalReranker(model_name, *, model=None)`; `CohereReranker(api_key, model, *, client=None)`; `build_reranker(settings, *, local=None, cohere=None) -> Reranker`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_rerank.py`:

```python
from __future__ import annotations

from app.config import Settings
from app.rag.rerank import CohereReranker, LocalReranker, build_reranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.seen = None
    def predict(self, pairs):
        self.seen = pairs
        return self.scores


def test_local_reranker_orders_by_score_and_truncates():
    ce = FakeCrossEncoder([0.1, 0.9, 0.5])
    idx = LocalReranker("m", model=ce).rerank("q", ["a", "b", "c"], top_n=2)
    assert idx == [1, 2]  # 0.9(b) > 0.5(c) > 0.1(a); top 2 -> indices 1, 2
    assert ce.seen == [("q", "a"), ("q", "b"), ("q", "c")]


def test_local_reranker_empty_docs():
    assert LocalReranker("m", model=FakeCrossEncoder([])).rerank("q", [], 5) == []


class _R:
    def __init__(self, i):
        self.index = i

class FakeCohereClient:
    def __init__(self, idxs):
        self.idxs = idxs
        self.called = None
    def rerank(self, query, documents, top_n, model):
        self.called = (query, documents, top_n, model)
        return type("Res", (), {"results": [_R(i) for i in self.idxs[:top_n]]})()


def test_cohere_reranker_maps_result_indices():
    c = FakeCohereClient([2, 0, 1])
    out = CohereReranker("key", "rerank-english-v3.0", client=c).rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [2, 0]
    assert c.called[3] == "rerank-english-v3.0"


def test_build_reranker_selects_provider():
    local, coh = object(), object()
    assert build_reranker(Settings(rerank_provider="local"), local=local, cohere=coh) is local
    assert build_reranker(Settings(rerank_provider="cohere"), local=local, cohere=coh) is coh


def test_build_reranker_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        build_reranker(Settings(rerank_provider="bogus"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_rerank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.rerank'`

- [ ] **Step 3: Implement**

Create `ml/app/rag/rerank.py`:

```python
from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]: ...


class LocalReranker:
    """Cross-encoder reranker via sentence-transformers. Inject `model` in
    tests to avoid loading a real CrossEncoder."""

    def __init__(self, model_name: str, *, model=None) -> None:
        self.model_name = model_name
        self._model = model

    def _m(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]:
        if not docs:
            return []
        scores = self._m().predict([(query, d) for d in docs])
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        return order[:top_n]


class CohereReranker:
    """Cohere Rerank. Inject `client` in tests to avoid the SDK/network."""

    def __init__(self, api_key: str, model: str, *, client=None) -> None:
        self.model = model
        if client is None:
            from cohere import Client  # lazy

            client = Client(api_key=api_key)
        self.client = client

    def rerank(self, query: str, docs: list[str], top_n: int) -> list[int]:
        if not docs:
            return []
        res = self.client.rerank(query=query, documents=docs, top_n=top_n, model=self.model)
        return [r.index for r in res.results]


def build_reranker(settings, *, local=None, cohere=None):
    provider = settings.rerank_provider.lower()
    if provider == "local":
        return local or LocalReranker(settings.rerank_model_local)
    if provider == "cohere":
        return cohere or CohereReranker(settings.cohere_api_key, settings.cohere_rerank_model)
    raise ValueError(f"unknown rerank_provider: {settings.rerank_provider!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_rerank.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/rerank.py tests/unit/test_rerank.py
git commit -m "feat(ml): add pluggable reranker (local cross-encoder default, cohere hatch)"
```

---

### Task 5: Sparse support in the vector store

Add optional sparse to `Record`, `upsert`, and `query`, keeping the dense-only path byte-for-byte unchanged.

**Files:**
- Modify: `ml/app/rag/vectorstore.py`
- Test: `ml/tests/unit/test_vectorstore.py`

**Interfaces:**
- Produces: `Record` gains `sparse_values: Optional[dict] = None`. `PineconeStore.upsert` adds a `sparse_values` key to a vector dict only when present. `PineconeStore.query(dense, top_k, flt, namespace, sparse=None)` passes `sparse_vector=sparse` to the index only when `sparse is not None`. The `VectorStore` Protocol's `query` signature gains `sparse: Optional[dict] = None`.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_vectorstore.py`:

```python
class _SparseFakeIndex:
    def __init__(self):
        self.upserted = None
        self.queried = None
    def upsert(self, vectors, namespace):
        self.upserted = (vectors, namespace)
    def query(self, **kw):
        self.queried = kw
        return {"matches": [{"id": "d::s::0", "score": 1.0,
                             "metadata": {"source_doc_id": "d"}}]}


def test_upsert_includes_sparse_when_present():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).upsert(
        [Record(id="x", values=[0.1], sparse_values={"indices": [1], "values": [0.5]}, metadata={})],
        namespace="hybrid")
    assert idx.upserted[0][0]["sparse_values"] == {"indices": [1], "values": [0.5]}


def test_upsert_omits_sparse_when_none():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).upsert(
        [Record(id="x", values=[0.1], metadata={})], namespace="curated")
    assert "sparse_values" not in idx.upserted[0][0]


def test_query_passes_sparse_vector():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).query(
        [0.1], top_k=5, flt=None, namespace="hybrid", sparse={"indices": [1], "values": [0.5]})
    assert idx.queried["sparse_vector"] == {"indices": [1], "values": [0.5]}


def test_query_dense_only_omits_sparse_vector():
    idx = _SparseFakeIndex()
    PineconeStore("k", "i", index=idx).query([0.1], top_k=5, flt=None, namespace="curated")
    assert "sparse_vector" not in idx.queried
```

(If `test_vectorstore.py` does not already import `Record`/`PineconeStore`, they are in `app.rag.vectorstore`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_vectorstore.py::test_upsert_includes_sparse_when_present -v`
Expected: FAIL — `TypeError` on the unexpected `sparse_values=` kwarg to `Record`.

- [ ] **Step 3: Implement**

In `ml/app/rag/vectorstore.py`:

Add the field to `Record`:

```python
class Record(BaseModel):
    id: str
    values: list[float]
    metadata: dict
    sparse_values: Optional[dict] = None
```

Update `VectorStore` Protocol's `query`:

```python
    def query(
        self, dense: list[float], top_k: int, flt: Optional[dict], namespace: str,
        sparse: Optional[dict] = None,
    ) -> list[Match]: ...
```

Replace `PineconeStore.upsert`:

```python
    def upsert(self, records: list[Record], namespace: str) -> None:
        vectors = []
        for r in records:
            vec = {"id": r.id, "values": list(r.values), "metadata": r.metadata}
            if r.sparse_values is not None:
                vec["sparse_values"] = r.sparse_values
            vectors.append(vec)
        self._idx().upsert(vectors=vectors, namespace=namespace)
```

Replace `PineconeStore.query`'s signature and the index call (keep the match-parsing loop below it unchanged):

```python
    def query(
        self, dense: list[float], top_k: int, flt: Optional[dict], namespace: str,
        sparse: Optional[dict] = None,
    ) -> list[Match]:
        if sparse is not None:
            res = self._idx().query(
                vector=list(dense), sparse_vector=sparse, top_k=top_k,
                include_metadata=True, filter=flt, namespace=namespace,
            )
        else:
            res = self._idx().query(
                vector=list(dense), top_k=top_k, include_metadata=True,
                filter=flt, namespace=namespace,
            )
        matches = res["matches"] if isinstance(res, dict) else res.matches
        out: list[Match] = []
        for m in matches:
            md = m["metadata"] if isinstance(m, dict) else m.metadata
            mid = m["id"] if isinstance(m, dict) else m.id
            score = m["score"] if isinstance(m, dict) else m.score
            out.append(Match(id=mid, score=float(score), metadata=dict(md or {})))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_vectorstore.py -v`
Expected: PASS — the four new tests plus all pre-existing vectorstore tests (dense path unchanged).

- [ ] **Step 5: Commit**

```bash
git add app/rag/vectorstore.py tests/unit/test_vectorstore.py
git commit -m "feat(ml): add sparse (hybrid) support to the vector store"
```

---

### Task 6: Extract `chunks_from_matches`

`DenseRagPipeline.retrieve` builds `Chunk`s from `Match` metadata inline; the hybrid retriever needs the identical logic. Extract it into a shared helper and rewire the dense pipeline (existing pipeline tests are the regression guard).

**Files:**
- Create: `ml/app/rag/retrieval.py`
- Modify: `ml/app/rag/pipeline.py`
- Test: `ml/tests/unit/test_retrieval.py` (create)

**Interfaces:**
- Produces: `chunks_from_matches(matches) -> list[Chunk]` in `app/rag/retrieval.py`, reading `chunk_text`/`source_doc_id`/`section`/`chunk_index` from each match's `.metadata` (defaults `""`/`""`/`""`/`0`).
- `DenseRagPipeline.retrieve` calls it; existing `tests/unit/test_pipeline.py` must stay green untouched.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_retrieval.py`:

```python
from __future__ import annotations

from app.rag.retrieval import chunks_from_matches
from app.rag.vectorstore import Match


def test_chunks_from_matches_reads_metadata():
    m = Match(id="d::s::2", score=1.0, metadata={
        "chunk_text": "t", "source_doc_id": "d", "section": "s",
        "chunk_index": 2, "severity": "high"})
    c = chunks_from_matches([m])[0]
    assert c.text == "t" and c.source_doc_id == "d" and c.section == "s"
    assert c.chunk_index == 2 and c.metadata["severity"] == "high"


def test_chunks_from_matches_defaults_missing_fields():
    c = chunks_from_matches([Match(id="x", score=0.0, metadata={})])[0]
    assert c.text == "" and c.source_doc_id == "" and c.chunk_index == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.retrieval'`

- [ ] **Step 3: Implement**

Create `ml/app/rag/retrieval.py`:

```python
from __future__ import annotations

from app.rag.models import Chunk


def chunks_from_matches(matches) -> list[Chunk]:
    """Rebuild Chunks from vector-store Matches. Retrieval stores the chunk
    text and provenance in metadata, so every retriever reconstructs Chunks
    the same way."""
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
```

In `ml/app/rag/pipeline.py`, add the import and replace the body of `DenseRagPipeline.retrieve`:

```python
from app.rag.retrieval import chunks_from_matches
```

```python
    def retrieve(self, query: str, k: int) -> list[Chunk]:
        vec = self.embedder.embed_query(query)
        matches = self.store.query(
            vec.tolist(), top_k=k, flt=None, namespace=self.namespace
        )
        return chunks_from_matches(matches)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_retrieval.py tests/unit/test_pipeline.py -v`
Expected: PASS — new helper tests plus the unchanged dense pipeline tests.

- [ ] **Step 5: Commit**

```bash
git add app/rag/retrieval.py app/rag/pipeline.py tests/unit/test_retrieval.py
git commit -m "refactor(ml): extract chunks_from_matches shared by dense + hybrid"
```

---

### Task 7: `HybridRerankRetriever` + convex scaling

**Files:**
- Create: `ml/app/rag/hybrid.py`
- Test: `ml/tests/unit/test_hybrid.py`

**Interfaces:**
- Consumes: `Embedder.embed_query`, `SparseEncoder.encode_query`, `VectorStore.query(..., sparse=)`, `Reranker.rerank`, `chunks_from_matches`.
- Produces: `convex_scale(dense: list[float], sparse: dict, alpha: float) -> tuple[list[float], dict]` (dense × alpha; sparse values × (1-alpha)); `HybridRerankRetriever(embedder, sparse_encoder, store, namespace, *, reranker=None, alpha=0.5, top_k=50)` with `retrieve(query, k) -> list[Chunk]`.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_hybrid.py`:

```python
from __future__ import annotations

import numpy as np

from app.rag.hybrid import HybridRerankRetriever, convex_scale
from app.rag.vectorstore import Match


class FakeEmbedder:
    dimension = 2
    def embed(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])
    def embed_query(self, text):
        return np.array([1.0, 1.0])


class FakeSparse:
    def encode_query(self, text):
        return {"indices": [3], "values": [2.0]}


class FakeStore:
    def __init__(self):
        self.q = None
    def query(self, dense, top_k, flt, namespace, sparse=None):
        self.q = dict(dense=dense, top_k=top_k, namespace=namespace, sparse=sparse)
        return [Match(id=f"d{i}::s::0", score=1.0,
                      metadata={"chunk_text": f"t{i}", "source_doc_id": f"d{i}",
                                "section": "s", "chunk_index": 0}) for i in range(3)]


class ReverseReranker:
    def rerank(self, query, docs, top_n):
        return list(range(len(docs)))[::-1][:top_n]


def test_convex_scale_weights_both_sides():
    d, s = convex_scale([1.0, 1.0], {"indices": [3], "values": [2.0]}, 0.75)
    assert d == [0.75, 0.75]
    assert s == {"indices": [3], "values": [0.5]}  # 2.0 * (1 - 0.75)


def test_retrieve_queries_top_k_scales_sparse_and_reranks():
    store = FakeStore()
    r = HybridRerankRetriever(FakeEmbedder(), FakeSparse(), store, "hybrid",
                              reranker=ReverseReranker(), alpha=0.5, top_k=50)
    chunks = r.retrieve("q", 2)
    assert store.q["top_k"] == 50 and store.q["namespace"] == "hybrid"
    assert store.q["sparse"] == {"indices": [3], "values": [1.0]}  # 2.0 * 0.5
    assert [c.source_doc_id for c in chunks] == ["d2", "d1"]  # reversed, top 2


def test_retrieve_without_reranker_truncates_in_store_order():
    store = FakeStore()
    r = HybridRerankRetriever(FakeEmbedder(), FakeSparse(), store, "hybrid",
                              reranker=None, alpha=0.5, top_k=50)
    assert [c.source_doc_id for c in r.retrieve("q", 2)] == ["d0", "d1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.hybrid'`

- [ ] **Step 3: Implement**

Create `ml/app/rag/hybrid.py`:

```python
from __future__ import annotations

from app.rag.models import Chunk
from app.rag.retrieval import chunks_from_matches


def convex_scale(dense: list[float], sparse: dict, alpha: float) -> tuple:
    """Weight dense vs sparse for a Pinecone hybrid query: dense * alpha,
    sparse values * (1 - alpha). Embeddings are assumed dot-product scored."""
    scaled_dense = [float(v) * alpha for v in dense]
    scaled_sparse = {
        "indices": list(sparse["indices"]),
        "values": [float(v) * (1.0 - alpha) for v in sparse["values"]],
    }
    return scaled_dense, scaled_sparse


class HybridRerankRetriever:
    """Two-stage retrieval: hybrid (dense + BM25 sparse) recall of top_k, then
    optional cross-encoder rerank to the final k. Retrieval-stage only —
    exposes `.retrieve` like the dense pipeline."""

    def __init__(self, embedder, sparse_encoder, store, namespace, *,
                 reranker=None, alpha: float = 0.5, top_k: int = 50) -> None:
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.store = store
        self.namespace = namespace
        self.reranker = reranker
        self.alpha = alpha
        self.top_k = top_k

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        dense = self.embedder.embed_query(query)
        sparse = self.sparse_encoder.encode_query(query)
        scaled_dense, scaled_sparse = convex_scale(list(dense), sparse, self.alpha)
        matches = self.store.query(
            scaled_dense, top_k=self.top_k, flt=None,
            namespace=self.namespace, sparse=scaled_sparse,
        )
        cands = chunks_from_matches(matches)
        if self.reranker is None:
            return cands[:k]
        order = self.reranker.rerank(query, [c.text for c in cands], top_n=k)
        return [cands[i] for i in order]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_hybrid.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/hybrid.py tests/unit/test_hybrid.py
git commit -m "feat(ml): add hybrid+rerank retriever with convex dense/sparse scaling"
```

---

### Task 8: Optional sparse in `build_records`

Extend the existing builder so the same function produces dense-only records (default) or hybrid records (dense + sparse) — no duplicate builder.

**Files:**
- Modify: `ml/app/ingest/build_index.py`
- Test: `ml/tests/unit/test_build_index.py`

**Interfaces:**
- Produces: `build_records(docs, chunker, embedder, sparse_encoder=None) -> list[Record]`. When `sparse_encoder` is given, it `fit`s on the chunk texts and attaches per-chunk `sparse_values`; when `None`, `sparse_values` stays `None` (existing dense behavior, unchanged).

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_build_index.py` (reuse the file's existing fake chunker/embedder/`docs` fixtures; if it lacks them, construct a one-section `Monograph`, a chunker whose `.chunk` returns one `Chunk`, and an embedder whose `.embed` returns a numpy array of one row):

```python
class _FakeSparseEnc:
    def __init__(self):
        self.fitted = None
    def fit(self, texts):
        self.fitted = list(texts)
    def encode_documents(self, texts):
        return [{"indices": [1], "values": [0.5]} for _ in texts]


def test_build_records_attaches_sparse_when_encoder_given():
    from app.ingest.build_index import build_records
    enc = _FakeSparseEnc()
    recs = build_records(_docs(), _chunker(), _embedder(), sparse_encoder=enc)
    assert enc.fitted is not None
    assert recs and all(r.sparse_values == {"indices": [1], "values": [0.5]} for r in recs)


def test_build_records_sparse_none_by_default():
    from app.ingest.build_index import build_records
    recs = build_records(_docs(), _chunker(), _embedder())
    assert recs and all(r.sparse_values is None for r in recs)
```

(Use whatever the existing test file already names its fakes; `_docs()`/`_chunker()`/`_embedder()` above stand in for them.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_build_index.py::test_build_records_attaches_sparse_when_encoder_given -v`
Expected: FAIL — `TypeError: build_records() got an unexpected keyword argument 'sparse_encoder'`

- [ ] **Step 3: Implement**

Replace `build_records` in `ml/app/ingest/build_index.py`:

```python
def build_records(
    docs: list[Monograph], chunker: Chunker, embedder: Embedder, sparse_encoder=None
) -> list[Record]:
    chunks = [c for doc in docs for c in chunker.chunk(doc)]
    if not chunks:
        return []
    texts = [c.text for c in chunks]
    vectors = embedder.embed(texts)
    sparses = None
    if sparse_encoder is not None:
        sparse_encoder.fit(texts)
        sparses = sparse_encoder.encode_documents(texts)
    records: list[Record] = []
    for i, (c, vec) in enumerate(zip(chunks, vectors)):
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
                sparse_values=(sparses[i] if sparses is not None else None),
                metadata=metadata,
            )
        )
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_build_index.py -v`
Expected: PASS — new sparse tests plus the pre-existing dense build_records tests (default path unchanged).

- [ ] **Step 5: Commit**

```bash
git add app/ingest/build_index.py tests/unit/test_build_index.py
git commit -m "feat(ml): build_records optionally attaches BM25 sparse vectors"
```

---

### Task 9: Retrieval experiment harness

**Files:**
- Create: `ml/scripts/run_retrieval_experiment.py`
- Test: `ml/tests/unit/test_run_retrieval_experiment.py`

**Interfaces:**
- Consumes: `evaluate_retrieval`/`compare_strategies`/`pick_winner`/`to_markdown` (Task 1), `build_records` (Task 8), `BGEEmbedder`, `BM25SparseEncoder`, `PineconeStore`, `HybridRerankRetriever`, `build_reranker`, `load_corpus`, `load_queries`, `FixedSizeChunker`.
- Produces: `pick_best_alpha(sweep: dict) -> float` (best alpha key by nDCG, recall tie-break); `main()`. `main()` is the live run (deferred); only `pick_best_alpha` is unit-tested.

- [ ] **Step 1: Write the failing test**

Create `ml/tests/unit/test_run_retrieval_experiment.py`:

```python
from __future__ import annotations

from scripts.run_retrieval_experiment import pick_best_alpha


def test_pick_best_alpha_by_ndcg_recall_tiebreak():
    sweep = {
        0.0: {"ndcg": 0.50, "recall_at_k": 0.40},
        0.5: {"ndcg": 0.72, "recall_at_k": 0.66},
        1.0: {"ndcg": 0.72, "recall_at_k": 0.55},
    }
    assert pick_best_alpha(sweep) == 0.5  # ties on ndcg (0.72) broken by recall
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_run_retrieval_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_retrieval_experiment'`

- [ ] **Step 3: Implement**

Create `ml/scripts/run_retrieval_experiment.py`:

```python
from __future__ import annotations

import json
import os

from app.config import get_settings
from app.eval.dataset import load_queries
from app.eval.retrieval_experiment import (
    compare_strategies, evaluate_retrieval, pick_winner, to_markdown,
)
from app.ingest.build_index import build_records
from app.rag.chunking import FixedSizeChunker
from app.rag.corpus import load_corpus
from app.rag.embeddings import BGEEmbedder
from app.rag.hybrid import HybridRerankRetriever
from app.rag.rerank import build_reranker
from app.rag.sparse import BM25SparseEncoder
from app.rag.vectorstore import PineconeStore


def pick_best_alpha(sweep: dict) -> float:
    """The alpha whose hybrid config maximizes nDCG (recall tie-break)."""
    return pick_winner(sweep)


def main() -> None:
    settings = get_settings()
    queries = load_queries(os.path.join(settings.eval_dir, "queries.json"))
    docs = load_corpus(settings.corpus_dir)
    embedder = BGEEmbedder(settings.embedding_model, settings.embedding_dim)
    sparse = BM25SparseEncoder()
    store = PineconeStore(settings.pinecone_api_key, settings.pinecone_index)
    store.ensure_index(settings.embedding_dim, settings.pinecone_metric,
                       settings.pinecone_cloud, settings.pinecone_region)

    # Build the hybrid namespace (dense + BM25 sparse); persist BM25 params.
    records = build_records(
        docs, FixedSizeChunker(chunk_size=512, overlap=0), embedder, sparse_encoder=sparse
    )
    store.upsert(records, namespace=settings.hybrid_namespace)
    sparse.dump(settings.bm25_params_path)
    print(f"indexed {len(records)} hybrid chunks -> namespace '{settings.hybrid_namespace}'")

    # Alpha sweep, retrieval-only (rerank off). alpha=1.0 is dense-only,
    # alpha=0.0 is sparse-only.
    sweep = {}
    for a in settings.hybrid_alphas:
        retriever = HybridRerankRetriever(
            embedder, sparse, store, settings.hybrid_namespace,
            reranker=None, alpha=a, top_k=settings.hybrid_top_k,
        )
        sweep[a] = evaluate_retrieval(retriever, queries, k=5)
        print(f"alpha={a}: {sweep[a]}")
    best_alpha = pick_best_alpha(sweep)

    # Hybrid(best alpha) + rerank on.
    reranker = build_reranker(settings)
    hybrid_rerank = HybridRerankRetriever(
        embedder, sparse, store, settings.hybrid_namespace,
        reranker=reranker, alpha=best_alpha, top_k=settings.hybrid_top_k,
    )

    results = {f"hybrid(alpha={a})": agg for a, agg in sweep.items()}
    results[f"hybrid(alpha={best_alpha})+rerank"] = evaluate_retrieval(hybrid_rerank, queries, k=5)

    rows = compare_strategies(results)
    winner = pick_winner(results)
    os.makedirs(settings.reports_dir, exist_ok=True)
    with open(os.path.join(settings.reports_dir, "retrieval.json"), "w", encoding="utf-8") as fh:
        json.dump({"results": results, "winner": winner, "best_alpha": best_alpha}, fh, indent=2)
    md = to_markdown(rows, winner)
    with open(os.path.join(settings.reports_dir, "retrieval.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_run_retrieval_experiment.py -v`
Expected: PASS

- [ ] **Step 5: Run the full offline suite**

Run: `python -m pytest tests/unit tests/eval -q`
Expected: PASS (all offline tests green; no new integration tests added this phase).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_retrieval_experiment.py tests/unit/test_run_retrieval_experiment.py
git commit -m "feat(ml): add retrieval experiment harness (dense/hybrid-sweep/rerank)"
```

---

## Operator run book (provisioned machine, after all 9 tasks)

Phase 1b must already have been run once (deps installed, `PINECONE_API_KEY` set).

```bash
cd ml && source venv/bin/activate
git pull
pip install -r requirements.txt            # pulls pinecone-text + cohere
python -m pytest tests/unit tests/eval -q   # offline gate

# runs the whole experiment: builds the `hybrid` namespace, sweeps alpha,
# reranks the best, writes eval/reports/retrieval.{json,md}
python -m scripts.run_retrieval_experiment
```

`bge-reranker-base` (~1GB) downloads on first run and runs on CPU. To use Cohere instead: set `RERANK_PROVIDER=cohere` and `COHERE_API_KEY=...`.

**Alpha endpoints:** `alpha=1.0` weights sparse to zero (dense-only) and `alpha=0.0` weights dense to zero (sparse-only). If the live Pinecone index rejects an all-zero vector at an extreme, drop that endpoint from `hybrid_alphas` (env/config) and re-run — the interior blend is what matters.

**Adopting the winner (manual follow-up):** once `retrieval.md` names the winning config, wire it into the answer path by replacing the `DenseRagPipeline` construction in `scripts/run_dense.py` with a `HybridRerankRetriever` at the winning alpha (and `build_reranker(settings)` if rerank won), pointed at the `hybrid` namespace, and load BM25 params from `settings.bm25_params_path`. Record the precision@5 lift in the README debrief (Phase 5).

**If the result is flat** (hybrid/rerank within noise of dense): document it honestly rather than manufacturing a winner (design spec §11 / §8).

---

## Self-Review

- **Spec coverage (design §4–§6):** `sparse.py` (Task 3), `rerank.py` pluggable factory (Task 4), vectorstore sparse (Task 5), `hybrid.py` retriever + convex scale (Task 7), hybrid index build (Task 8), retrieval-only harness with alpha sweep + rerank (Task 9). Shared-helper reuse via extractions (Tasks 1, 6) satisfies the "reuse, do not duplicate" constraint. Live run + winner adoption deferred to the run book (spec §2, §9).
- **Placeholder scan:** none — every code step carries complete, runnable code, except Tasks 1/8 which deliberately say "move verbatim" / "reuse the file's existing fakes" (repeating the moved bodies or re-deriving existing fixtures would risk drift from the actual committed code).
- **Type consistency:** `SparseEncoder.encode_query -> dict`, `Reranker.rerank(query, docs, top_n) -> list[int]`, `VectorStore.query(dense, top_k, flt, namespace, sparse=None)`, `Record.sparse_values: Optional[dict]`, `convex_scale(dense, sparse, alpha) -> (list, dict)`, `HybridRerankRetriever(embedder, sparse_encoder, store, namespace, *, reranker, alpha, top_k)`, `build_records(docs, chunker, embedder, sparse_encoder=None)`, `chunks_from_matches(matches) -> list[Chunk]`, and the shared harness signatures are used identically across tasks. `build_reranker` mirrors `build_judge`'s injectable-factory shape.
```
