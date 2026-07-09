# ClearRx — Live Runs Provisioning Checklist

Everything the deferred live experiments need, in order. These produce the real metrics for the Phase 5 README debrief. All unit/offline tests already pass without any of this; this checklist is only for the **live** runs (Pinecone + Ollama + model downloads).

> **Config reads OS env vars, not `.env`.** `ml/app/config.py` sets `env_file=None`, so pydantic-settings reads the **process environment**. Export the vars in your shell (§2) before running any `python -m ...` command below — putting them in `.env` will NOT reach the ML config.

---

## 0. Accounts & installs (one-time)

- [ ] **Pinecone** free-tier account → create an API key. That's all — the index is created automatically (serverless, `aws`/`us-east-1`, `dotproduct` metric) by the build step.
- [ ] **Ollama** installed and running locally (`https://ollama.com`). Default endpoint `http://localhost:11434`.
- [ ] **(Optional) Cohere** API key — only if you want to compare the hosted reranker in Phase 3 (`RERANK_PROVIDER=cohere`). The default reranker is local `bge-reranker-base`, so this is optional.
- [ ] **Disk:** ~7 GB free for model downloads (BGE ~1.3 GB, bge-reranker ~1 GB, llama3.1 ~4.7 GB).

## 1. Install deps + pull models (one-time)

```bash
cd ml && source venv/bin/activate
pip install -r requirements.txt          # pulls pinecone, pinecone-text, cohere, sentence-transformers, langchain-text-splitters, ...
ollama pull llama3.1                      # generation + judge model (~4.7 GB)
```
BGE (`BAAI/bge-large-en-v1.5`) and `bge-reranker-base` download automatically on first use via sentence-transformers.

## 2. Export environment (every shell session that runs a live command)

```bash
export PINECONE_API_KEY="pc-..."          # required
# defaults below already match config.py — override only if needed:
# export OLLAMA_BASE_URL="http://localhost:11434"
# export GEN_MODEL="llama3.1"
# export JUDGE_MODEL="llama3.1"
# export COHERE_API_KEY="..."             # only for the Cohere rerank comparison
```
Sanity check the config picks it up:
```bash
python -c "from app.config import get_settings; s=get_settings(); print('pinecone key set:', bool(s.pinecone_api_key), '| ollama:', s.ollama_base_url)"
```

## 3. Build the dense index (prerequisite for Phase 1b + Phase 4 streaming)

```bash
python -m app.ingest.build_index         # corpus -> chunk -> BGE embed -> upsert to the `curated` namespace
```
This creates the Pinecone index and populates the `curated` namespace that `run_dense.py` and the live `/query` service read.

## 4. Phase 1b — dense retrieval eval + judge calibration

```bash
python -m scripts.run_dense
```
Produces `ml/eval/reports/dense.{json,md}`, prints the **keyword→dense** metric deltas and the **judge-vs-human agreement %**. If agreement < 80%, rerun with a stronger judge:
```bash
JUDGE_PROVIDER=openai JUDGE_MODEL=gpt-4o-mini OPENAI_API_KEY=sk-... python -m scripts.run_dense
```

## 5. Phase 2 — chunking experiment

```bash
pip install -r requirements.txt          # ensure langchain-text-splitters present (already in step 1)
python -m scripts.run_chunking_experiment
```
Re-indexes each strategy into `chunk_fixed` / `chunk_recursive` / `chunk_semantic`, retrieval-evals all three, writes `ml/eval/reports/chunking.{json,md}` + the winner (nDCG@5, recall tie-break). The `curated` namespace is untouched.

## 6. Phase 3 — hybrid + rerank experiment

```bash
python -m scripts.run_retrieval_experiment
```
Builds the `hybrid` namespace (dense + BM25 sparse), sweeps alpha, reranks the best with the **local `bge-reranker-base`** (downloads on first run), writes `ml/eval/reports/retrieval.{json,md}` + `bm25_params.json`. To try Cohere instead: `RERANK_PROVIDER=cohere COHERE_API_KEY=... python -m scripts.run_retrieval_experiment`.
- **Alpha endpoints caveat:** if Pinecone rejects an all-zero vector at `alpha=0.0`/`1.0`, drop those endpoints from `hybrid_alphas` and rerun (the interior blend is what matters).
- **Winner adoption is manual** and deferred (see the Phase 3 plan run book): to serve the winner, point the pipeline at the `hybrid` namespace + load `bm25_params.json`, and validate a real BM25 dump→load round-trip first.

## 7. Phase 4 — end-to-end streaming smoke (browser → Express → ML → Ollama)

Three terminals (Ollama running, `curated` index built from §3):
```bash
# 1. ML service
cd ml && source venv/bin/activate && PINECONE_API_KEY=pc-... uvicorn app.main:app --port 8000

# 2. Express proxy
cd api && ML_BASE=http://localhost:8000 npm start          # :3001

# 3. Frontend
cd frontend && VITE_API_BASE_URL=http://localhost:3001 npm run dev   # :8080
```
Open the dashboard, type a drug-interaction question in the **Ask ClearRx** panel, confirm tokens render live with a Sources list. (This path uses the **dense** `curated` retrieval + Ollama; adopting the Phase 3 hybrid winner into this path is the deferred manual step in §6.)

## 8. (Optional) Run the gated integration tests

With Pinecone key + Ollama live:
```bash
cd ml && source venv/bin/activate && RUN_INTEGRATION=1 PINECONE_API_KEY=pc-... python -m pytest -m integration -q
```

---

## What each run produces → Phase 5 README

| Run | Report | Feeds the README table row |
|---|---|---|
| `run_baseline` (already done, offline) | `eval/reports/baseline.{json,md}` | keyword baseline (starting line) |
| `run_dense` (§4) | `eval/reports/dense.{json,md}` | keyword → BGE dense; judge calibration % |
| `run_chunking_experiment` (§5) | `eval/reports/chunking.{json,md}` | chunking-strategy comparison + winner |
| `run_retrieval_experiment` (§6) | `eval/reports/retrieval.{json,md}` | dense → hybrid(α-swept) → hybrid+rerank precision@5 lift |
| §7 streaming smoke | — | qualitative: streamed answer == non-stream, live UX |

Commit the generated `eval/reports/*.json,md` after each run — they are the evidence the **Phase 5 debrief** turns into the baseline→final metrics table and per-change impact narrative.
