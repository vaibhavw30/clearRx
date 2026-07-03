# ClearRx Phase 1a — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the ClearRx eval foundation from 8 monographs / 13 queries to ≥25 monographs / ≥40 queries (including negative "no-interaction" cases), make the eval runner aggregate retrieval metrics honestly in the presence of ungradable queries, and re-capture the keyword baseline as the new higher-signal starting line.

**Architecture:** Pure offline data + eval-harness work. No new external services. Two small code changes (dataset query-type extension, runner aggregation) precede the two content-authoring tasks (corpus, queries+labels), and a final re-run resets the committed baseline report. Everything stays behind the interfaces already built in Phase 0.

**Tech Stack:** Python 3.9.6, pydantic v2, pytest. All tests offline/fakes-only.

## Global Constraints

- Python **3.9.6** — every module starts with `from __future__ import annotations`.
- **pydantic v2** for all models.
- All work runs from the `ml/` directory with the venv active: `cd ml && source venv/bin/activate`.
- Tests are **offline, fakes-only**. No network, no model downloads, no API keys in this phase.
- Scripts use repo-relative paths from `ml/`: corpus at `data/corpus`, queries at `data/eval/queries.json`, reports at `eval/reports`.
- Run the full unit + eval suite with `python -m pytest -q` (from `ml/`). It must stay green after every task.
- Commit after every task with a `feat(ml):` or `test(ml):` / `data(ml):` prefixed message.
- Clinical content must be plausibly correct and **cited** — every monograph carries ≥1 real `evidence` entry with a real URL. This is a patient-safety project; wrong facts defeat its purpose.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `ml/app/eval/dataset.py` | Eval query schema + loader | Modify: add `no_interaction` query type |
| `ml/app/eval/runner.py` | Run eval, aggregate metrics | Modify: gate ranking/coverage metrics on gradability |
| `ml/tests/unit/test_dataset.py` | Dataset unit tests | Modify: add no-interaction case |
| `ml/tests/unit/test_runner.py` | Runner unit tests | Modify: add negative-query aggregation case |
| `ml/data/corpus/int_*.json` | Interaction monographs | Create: ≥17 new files |
| `ml/data/eval/queries.json` | Labeled eval queries | Modify: grow to ≥40 |
| `ml/data/eval/human_labels.json` | Judge-calibration labels | Modify: grow to ≥15 |
| `ml/tests/eval/test_seed_data.py` | Corpus+query integrity | Modify: raise thresholds, add integrity checks |
| `ml/eval/reports/baseline.{json,md}` | Committed baseline | Regenerate |

---

### Task 1: Support "no-interaction" query type

Negative queries (drug pairs with no clinically significant interaction) are needed so precision is actually testable, but the current `EvalQuery` validator rejects any `query_type` outside `{interaction, dosage, contraindication}`. Add a fourth type.

**Files:**
- Modify: `ml/app/eval/dataset.py:7` (the `_TYPES` set)
- Test: `ml/tests/unit/test_dataset.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `EvalQuery` now accepts `query_type == "no_interaction"`. Such queries carry `expected_doc_ids: []` (no gold doc) and rely on `expected_answer_facts` + `must_not_say` for grading.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_dataset.py`:

```python
def test_no_interaction_query_type_is_accepted(tmp_path):
    from app.eval.dataset import load_queries
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"queries": [{
        "id": "n001",
        "query": "Can I take amoxicillin with acetaminophen?",
        "query_type": "no_interaction",
        "expected_doc_ids": [],
        "expected_retrieval_topics": [],
        "expected_answer_facts": ["No clinically significant interaction"],
        "must_not_say": ["dangerous interaction", "avoid this combination"],
        "severity": "low",
    }]}))
    queries = load_queries(str(p))
    assert queries[0].query_type == "no_interaction"
    assert queries[0].expected_doc_ids == []
```

(If `test_dataset.py` does not already `import json`, add it at the top.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_dataset.py::test_no_interaction_query_type_is_accepted -v`
Expected: FAIL — `DatasetError: ... query_type must be one of ['contraindication', 'dosage', 'interaction']`

- [ ] **Step 3: Write minimal implementation**

In `ml/app/eval/dataset.py`, change line 7:

```python
_TYPES = {"interaction", "dosage", "contraindication", "no_interaction"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_dataset.py -v`
Expected: PASS (all dataset tests, including the new one)

- [ ] **Step 5: Commit**

```bash
git add app/eval/dataset.py tests/unit/test_dataset.py
git commit -m "feat(ml): accept no_interaction eval query type"
```

---

### Task 2: Gate ranking/coverage metrics on gradability

A `no_interaction` query has an empty `expected_doc_ids`, so `precision_at_k`/`recall_at_k`/`mrr`/`ndcg` are meaningless for it and currently return `0.0`, which would drag the aggregate down and hide real retrieval quality. Aggregate ranking metrics only over queries that have a gold doc; aggregate `retrieval_coverage` only over queries that declare topics; keep generation/safety/latency metrics over **all** queries.

**Files:**
- Modify: `ml/app/eval/runner.py` (the `run` method's per-row dict and the `aggregate` block)
- Test: `ml/tests/unit/test_runner.py`

**Interfaces:**
- Consumes: `EvalQuery.expected_doc_ids`, `EvalQuery.expected_retrieval_topics` (already exist).
- Produces: each per-query row gains `"retrieval_gradable": bool` and `"coverage_gradable": bool`; `aggregate` gains `"n_queries": float` and `"n_retrieval_gradable": float`. Ranking metrics in `aggregate` are means over gradable rows only. Existing keys and their meaning for all-positive query sets are unchanged.

- [ ] **Step 1: Write the failing test**

Add to `ml/tests/unit/test_runner.py` (reuses the existing `MultiChunkSameDocPipeline`, `FakeJudge`, `_query` helpers already in that file):

```python
def _negative_query():
    return EvalQuery(
        id="n001", query="amoxicillin acetaminophen", query_type="no_interaction",
        expected_doc_ids=[], expected_retrieval_topics=[],
        expected_answer_facts=["No clinically significant interaction"],
        must_not_say=["dangerous interaction"], severity="low",
    )


def test_negative_query_does_not_drag_ranking_metrics():
    # Pipeline always returns the warfarin doc; for the positive query that is
    # relevant (recall 1.0). The negative query has no gold doc and must be
    # excluded from ranking aggregates rather than counted as a 0.0.
    runner = EvalRunner(MultiChunkSameDocPipeline(), FakeJudge(), k=5, clock=lambda: 0.0)
    report = runner.run([_query(), _negative_query()])
    agg = report.aggregate
    assert agg["n_queries"] == 2.0
    assert agg["n_retrieval_gradable"] == 1.0
    assert agg["recall_at_k"] == 1.0   # averaged over the 1 gradable query, not 0.5
    assert agg["mrr"] == 1.0
    # per-query rows expose gradability
    rows = {r["id"]: r for r in report.per_query}
    assert rows["q001"]["retrieval_gradable"] is True
    assert rows["n001"]["retrieval_gradable"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_runner.py::test_negative_query_does_not_drag_ranking_metrics -v`
Expected: FAIL — `KeyError: 'n_queries'` (and, once that is added, `recall_at_k == 0.5`)

- [ ] **Step 3: Write minimal implementation**

In `ml/app/eval/runner.py`, add a gradability-aware mean helper next to `_mean`:

```python
def _mean_where(rows: list[dict], key: str, gate: str) -> float:
    vals = [r[key] for r in rows if r[gate]]
    return sum(vals) / len(vals) if vals else 0.0
```

In `run()`, add two flags to each appended row (inside the `rows.append({...})` dict):

```python
                    "retrieval_gradable": bool(relevant),
                    "coverage_gradable": bool(q.expected_retrieval_topics),
```

Then replace the `aggregate` dict so ranking/coverage metrics use `_mean_where`, generation/latency stay over all rows, and the counts are exposed:

```python
        aggregate = {
            "retrieval_coverage": _mean_where(rows, "retrieval_coverage", "coverage_gradable"),
            "precision_at_k": _mean_where(rows, "precision_at_k", "retrieval_gradable"),
            "recall_at_k": _mean_where(rows, "recall_at_k", "retrieval_gradable"),
            "mrr": _mean_where(rows, "mrr", "retrieval_gradable"),
            "ndcg": _mean_where(rows, "ndcg", "retrieval_gradable"),
            "fact_coverage": _mean([r["fact_coverage"] for r in rows]),
            "forbidden_violations": float(sum(r["forbidden_violations"] for r in rows)),
            "latency_ms_p50": percentile(latencies, 50),
            "latency_ms_p95": percentile(latencies, 95),
            "n_queries": float(len(rows)),
            "n_retrieval_gradable": float(sum(1 for r in rows if r["retrieval_gradable"])),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_runner.py -v`
Expected: PASS — the new test plus all pre-existing runner tests (all-positive sets are unaffected: their gradable count equals their query count, so the means are identical).

- [ ] **Step 5: Commit**

```bash
git add app/eval/runner.py tests/unit/test_runner.py
git commit -m "feat(ml): gate ranking metrics on query gradability"
```

---

### Task 3: Expand the corpus to ≥25 monographs

Author ≥17 new interaction monographs so the corpus totals ≥25. Each new file is `ml/data/corpus/int_<druga>_<drugb>.json` and matches the `Monograph` schema exactly (`app/rag/models.py`). Use the existing `int_warfarin_ibuprofen.json` as the depth/quality exemplar: five sections (`summary`, `mechanism`, `clinical_effects`, `management`, `monitoring`), each 1–3 substantive sentences of original prose (do not copy source text verbatim), correct `severity`, real brand-name `aliases`, correct `drug_class_*`, and ≥1 real `evidence` citation with a working PubMed/label URL.

**Files:**
- Create: 17 files under `ml/data/corpus/` (list below)
- Test: `ml/tests/eval/test_seed_data.py`

**Interfaces:**
- Consumes: `Monograph` schema (`app/rag/models.py`) — unchanged.
- Produces: ≥25 loadable monographs whose ids are referenced by Task 4's queries.

**Interactions to author** (id = filename stem; all are well-documented, teachable pairs):

| id | drug_a / drug_b | classes | severity | clinical basis (one line) |
|---|---|---|---|---|
| `int_warfarin_amiodarone` | warfarin / amiodarone | anticoagulant / antiarrhythmic | high | CYP2C9 inhibition raises INR; reduce warfarin dose |
| `int_warfarin_fluconazole` | warfarin / fluconazole | anticoagulant / antifungal | high | Potent CYP2C9 inhibition, marked INR rise |
| `int_simvastatin_gemfibrozil` | simvastatin / gemfibrozil | statin / fibrate | high | Additive myopathy/rhabdomyolysis risk |
| `int_digoxin_verapamil` | digoxin / verapamil | cardiac glycoside / ccb | high | P-gp inhibition raises digoxin, bradycardia |
| `int_digoxin_amiodarone` | digoxin / amiodarone | cardiac glycoside / antiarrhythmic | high | Digoxin levels rise ~2x; halve dose |
| `int_methotrexate_trimethoprim` | methotrexate / trimethoprim | antimetabolite / antibiotic | high | Additive antifolate → marrow suppression |
| `int_lithium_hydrochlorothiazide` | lithium / hydrochlorothiazide | mood stabilizer / thiazide diuretic | high | Reduced renal Li clearance → toxicity |
| `int_lithium_ibuprofen` | lithium / ibuprofen | mood stabilizer / nsaid | moderate | NSAIDs raise serum lithium |
| `int_clopidogrel_omeprazole` | clopidogrel / omeprazole | antiplatelet / ppi | moderate | CYP2C19 inhibition blunts antiplatelet effect |
| `int_sertraline_tramadol` | sertraline / tramadol | ssri / opioid analgesic | high | Serotonin syndrome risk |
| `int_linezolid_sertraline` | linezolid / sertraline | antibiotic (MAOI) / ssri | high | Serotonin syndrome (MAO inhibition) |
| `int_ciprofloxacin_tizanidine` | ciprofloxacin / tizanidine | fluoroquinolone / muscle relaxant | high | CYP1A2 inhibition → hypotension (contraindicated) |
| `int_theophylline_ciprofloxacin` | theophylline / ciprofloxacin | bronchodilator / fluoroquinolone | high | CYP1A2 inhibition → theophylline toxicity |
| `int_carbamazepine_clarithromycin` | carbamazepine / clarithromycin | anticonvulsant / macrolide | high | CYP3A4 inhibition → carbamazepine toxicity |
| `int_allopurinol_azathioprine` | allopurinol / azathioprine | xanthine oxidase inhibitor / immunosuppressant | high | Blocked azathioprine metabolism → marrow toxicity |
| `int_rifampin_ethinylestradiol` | rifampin / ethinylestradiol | antibiotic (enzyme inducer) / oral contraceptive | moderate | CYP3A4 induction → contraceptive failure |
| `int_phenelzine_pseudoephedrine` | phenelzine / pseudoephedrine | maoi / sympathomimetic | high | Hypertensive crisis |
| `int_metoprolol_verapamil` | metoprolol / verapamil | beta blocker / ccb | high | Additive AV block / bradycardia |
| `int_sertraline_ibuprofen` | sertraline / ibuprofen | ssri / nsaid | moderate | Additive GI bleeding risk |

(19 listed; author at least 17. Skip any you cannot cite well and substitute another well-documented pair.)

Example of the target shape (write files at this depth):

```json
{
  "id": "int_warfarin_amiodarone",
  "drug_a": "warfarin",
  "drug_b": "amiodarone",
  "drug_a_aliases": ["coumadin", "jantoven"],
  "drug_b_aliases": ["cordarone", "pacerone"],
  "drug_class_a": "anticoagulant",
  "drug_class_b": "antiarrhythmic",
  "severity": "high",
  "sections": {
    "summary": "Amiodarone markedly potentiates warfarin, raising the INR and bleeding risk. The interaction develops over days to weeks and persists long after amiodarone is stopped.",
    "mechanism": "Amiodarone inhibits CYP2C9 and CYP3A4, the enzymes that clear warfarin's more potent S-enantiomer, reducing warfarin metabolism. Amiodarone's long half-life means the effect builds and resolves slowly.",
    "clinical_effects": "INR rises, often into a supratherapeutic range, with increased risk of bruising and major bleeding if the warfarin dose is not reduced.",
    "management": "Anticipate a warfarin dose reduction of roughly 30-50% when starting amiodarone, titrated to INR. Coordinate any amiodarone discontinuation with warfarin re-titration.",
    "monitoring": "Check INR more frequently (e.g., weekly) after starting or stopping amiodarone until stable, and counsel the patient on bleeding signs."
  },
  "evidence": [
    {"citation": "Sanoski CA, Bauman JL. Chest. 2002;121(1):19-23", "url": "https://pubmed.ncbi.nlm.nih.gov/11796427/"}
  ]
}
```

- [ ] **Step 1: Strengthen the integrity test first (fails)**

Replace the body of `test_seed_corpus_and_queries_load` in `ml/tests/eval/test_seed_data.py` and add a monograph-completeness test:

```python
def test_seed_corpus_and_queries_load():
    docs = load_corpus("data/corpus")
    queries = load_queries("data/eval/queries.json")
    doc_ids = {d.id for d in docs}
    assert len(docs) >= 25
    for q in queries:
        for did in q.expected_doc_ids:
            assert did in doc_ids, f"{q.id} references unknown doc {did}"


def test_every_monograph_is_complete_and_cited():
    docs = load_corpus("data/corpus")
    required = {"summary", "mechanism", "clinical_effects", "management", "monitoring"}
    for d in docs:
        assert required.issubset(d.sections), f"{d.id} missing sections"
        assert all(d.sections[s].strip() for s in required), f"{d.id} has an empty section"
        assert d.evidence, f"{d.id} has no evidence"
        assert all(e.url.startswith("http") for e in d.evidence), f"{d.id} evidence missing url"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_seed_data.py::test_seed_corpus_and_queries_load -v`
Expected: FAIL — `assert 8 >= 25`

- [ ] **Step 3: Author the monograph files**

Create the ≥17 JSON files from the table above at the exemplar's depth. Validate each loads as you go:

```bash
python -c "from app.rag.corpus import load_corpus; d=load_corpus('data/corpus'); print(len(d), 'monographs load')"
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_seed_data.py -v`
Expected: PASS (both integrity tests)

- [ ] **Step 5: Commit**

```bash
git add data/corpus/ tests/eval/test_seed_data.py
git commit -m "data(ml): expand interaction corpus to 25+ monographs"
```

---

### Task 4: Expand eval queries to ≥40 and human labels to ≥15

Grow `queries.json` to ≥40 labeled queries and `human_labels.json` to ≥15 labeled answers. The expanded set must deliberately exercise the failure modes the design cares about: brand↔generic name mismatches, dosage qualifiers, and negative (`no_interaction`) pairs so precision is testable.

**Files:**
- Modify: `ml/data/eval/queries.json`
- Modify: `ml/data/eval/human_labels.json`
- Test: `ml/tests/eval/test_seed_data.py`

**Interfaces:**
- Consumes: `EvalQuery` schema (Task 1, now including `no_interaction`); corpus doc ids from Task 3.
- Produces: ≥40 queries whose every `expected_doc_id` resolves to a Task 3 monograph; ≥15 human labels whose `human_fact_labels` length matches the referenced query's `expected_answer_facts` length.

**Composition targets** (aim for roughly):
- ≥8 `no_interaction` queries (empty `expected_doc_ids`, meaningful `must_not_say`).
- ≥8 queries phrased with **brand names only** (e.g., "Advil", "Coumadin", "Cordarone") whose gold doc uses the generic — tests alias resolution.
- ≥4 `dosage`-type queries.
- The rest `interaction` / `contraindication`, spread across the new monographs.

Query schema (all fields required as shown; `expected_doc_ids` empty only for `no_interaction`):

```json
{
  "id": "q014",
  "query": "Is it safe to take Cordarone with my Coumadin?",
  "query_type": "interaction",
  "expected_doc_ids": ["int_warfarin_amiodarone"],
  "expected_retrieval_topics": ["warfarin amiodarone interaction", "INR increase"],
  "expected_answer_facts": [
    "Amiodarone increases the INR",
    "Warfarin dose usually needs reduction",
    "Monitor INR more frequently"
  ],
  "must_not_say": ["no interaction", "safe to combine at normal doses"],
  "severity": "high"
}
```

Negative-query example:

```json
{
  "id": "n001",
  "query": "Can I take amoxicillin with acetaminophen?",
  "query_type": "no_interaction",
  "expected_doc_ids": [],
  "expected_retrieval_topics": [],
  "expected_answer_facts": ["No clinically significant interaction is expected"],
  "must_not_say": ["dangerous interaction", "avoid this combination", "serious bleeding"],
  "severity": "low"
}
```

Human-label entry (append to the `labels` array; `human_fact_labels` must be the same length and order as that query's `expected_answer_facts`):

```json
{
  "query_id": "q014",
  "answer": "Amiodarone raises your INR, so your warfarin dose usually has to be lowered and your INR checked more often.",
  "human_fact_labels": [true, true, true]
}
```

- [ ] **Step 1: Strengthen the seed-data test first (fails)**

Add to `ml/tests/eval/test_seed_data.py`:

```python
def test_query_set_size_and_shape():
    queries = load_queries("data/eval/queries.json")
    assert len(queries) >= 40
    by_type = {}
    for q in queries:
        by_type.setdefault(q.query_type, 0)
        by_type[q.query_type] += 1
    assert by_type.get("no_interaction", 0) >= 8, "need negative queries for precision"
    # every non-negative query names at least one gold doc
    for q in queries:
        if q.query_type != "no_interaction":
            assert q.expected_doc_ids, f"{q.id} has no gold doc"


def test_human_labels_align_with_queries():
    import json
    queries = {q.id: q for q in load_queries("data/eval/queries.json")}
    labels = json.load(open("data/eval/human_labels.json"))["labels"]
    assert len(labels) >= 15
    for lab in labels:
        q = queries[lab["query_id"]]
        assert len(lab["human_fact_labels"]) == len(q.expected_answer_facts), (
            f"{lab['query_id']} label length != expected_answer_facts length"
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_seed_data.py::test_query_set_size_and_shape -v`
Expected: FAIL — `assert 13 >= 40`

- [ ] **Step 3: Author the queries and labels**

Extend `queries.json` to ≥40 (keep ids unique; use `n###` for negatives, `q###` otherwise) and `human_labels.json` to ≥15 entries. Sanity-check as you go:

```bash
python -c "from app.eval.dataset import load_queries; print(len(load_queries('data/eval/queries.json')), 'queries load')"
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/eval/test_seed_data.py -v`
Expected: PASS (all four seed-data tests)

- [ ] **Step 5: Commit**

```bash
git add data/eval/queries.json data/eval/human_labels.json tests/eval/test_seed_data.py
git commit -m "data(ml): expand eval set to 40+ queries and 15+ human labels"
```

---

### Task 5: Re-capture the keyword baseline

Re-run the existing baseline runner over the expanded data and commit the regenerated report. This is the new, higher-signal starting line every Phase 1b/2/3 change will be measured against. `fact_coverage` stays `0.0` here — the baseline judge is the offline all-False stub; the judge becomes real in Phase 1b.

**Files:**
- Modify (regenerate): `ml/eval/reports/baseline.json`, `ml/eval/reports/baseline.md`

**Interfaces:**
- Consumes: `scripts/run_baseline.py` (unchanged), the expanded corpus + queries.
- Produces: committed baseline report reflecting ≥25 docs / ≥40 queries.

- [ ] **Step 1: Confirm the whole suite is green**

Run: `python -m pytest -q`
Expected: PASS (all unit + eval tests)

- [ ] **Step 2: Regenerate the baseline report**

Run: `python -m scripts.run_baseline`
Expected: prints a metrics table and `wrote eval/reports/baseline.json and eval/reports/baseline.md`. Confirm `n_queries` in `baseline.json`'s `aggregate` is ≥40 and `retrieval_coverage`/`recall_at_k` are now averaged over the larger gradable set.

- [ ] **Step 3: Eyeball the report for sanity**

Run: `python -c "import json; a=json.load(open('eval/reports/baseline.json'))['aggregate']; print(a)"`
Expected: `n_queries >= 40`, `n_retrieval_gradable` = (queries minus negatives), `0 < recall_at_k <= 1`, `fact_coverage == 0.0`.

- [ ] **Step 4: Commit**

```bash
git add eval/reports/baseline.json eval/reports/baseline.md
git commit -m "data(ml): reset keyword baseline over expanded eval set"
```

---

## Self-Review

- **Spec coverage:** Task 3 delivers the ≥25-monograph corpus decision; Task 4 delivers the ≥40-query / ≥15-label expansion incl. brand/dosage/negative failure modes (spec §5, §7); Tasks 1–2 make negative queries measurable without corrupting aggregates; Task 5 resets the committed baseline (spec §7 "honest starting line"). Local-stack / Pinecone / real-judge items are intentionally deferred to Phase 1b (below).
- **Placeholder scan:** No `TBD`/`TODO`. Content-authoring tasks (3, 4) specify exact filenames, a full exemplar, a quality rubric, and a validating test rather than inlining all prose — the prose is the deliverable, not a plan gap.
- **Type consistency:** `_mean_where(rows, key, gate)`, row keys `retrieval_gradable`/`coverage_gradable`, aggregate keys `n_queries`/`n_retrieval_gradable`, and `query_type == "no_interaction"` are used identically across Tasks 1, 2, and 4.

---

## Phase 1b roadmap (next plan file — do NOT execute here)

Per the project's empirical-phasing rule, Phase 1b gets its **own** full plan file authored after 1a lands and the reset baseline is visible. Sketch of what it will contain (local stack as chosen: BGE + Ollama + Pinecone):

1. **Config + deps + integration-skip plumbing** — add `pinecone_api_key`, `ollama_base_url` to `config.py`; add `pinecone-client` to `requirements.txt`; a `conftest.py` autoskip for `@pytest.mark.integration` when keys/services are absent.
2. **`Embedder` + `BGEEmbedder`** (`app/rag/embeddings.py`) — sentence-transformers `bge-large-en-v1.5`, `.dimension == 1024`; unit test with a fake, integration test asserts real shape/dim.
3. **`VectorStore` + `PineconeStore`** (`app/rag/vectorstore.py`) — dense upsert/query, `curated` namespace, metadata filter; `Record`/`Match` models; fake-client unit test + real round-trip integration test.
4. **`LLMClient` + `OllamaClient`** (`app/rag/generator.py`) — `generate` + `stream`; fake-HTTP unit test + real-Ollama integration test.
5. **`DenseRagPipeline`** (`app/rag/pipeline.py`) — embed→query→Chunks, context+citation prompt→generate; fully faked unit test.
6. **`ingest/build_index.py`** — corpus→chunk→embed→upsert CLI; unit-test the pure records builder.
7. **Real judge factory + wiring** — `OllamaClient.generate` as the `LLMJudge` callable, selected by `JUDGE_PROVIDER`/`JUDGE_MODEL`.
8. **Judge calibration** (`app/eval/calibration.py`) — score `human_labels.json`, report judge-vs-human agreement %.
9. **`scripts/run_dense.py`** — end-to-end dense run → `eval/reports/dense.{json,md}` + calibration, the keyword→BGE-dense comparison.

**User-provisioned prerequisites for 1b:** Pinecone free-tier API key; Ollama installed + `ollama pull llama3.1`; ~1.3 GB disk for the BGE model download.
