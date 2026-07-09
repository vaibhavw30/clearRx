from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.answer import build_interaction_response, citations_from_chunks, sse_frame
from app.config import get_settings
from app.deps import get_llm, get_retriever
from app.rag.corpus import load_corpus
from app.rag.pipeline import build_prompt
from app.schemas import (
    EnhancedInteractionRequest,
    HealthResponse,
    InteractionResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(title="ClearRx RAG service")

_NO_INFO = "No interaction information available in the corpus."


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", services={"rag": True})


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, retriever=Depends(get_retriever), llm=Depends(get_llm)) -> QueryResponse:
    chunks = retriever.retrieve(req.query, req.top_k)
    if not chunks:
        return QueryResponse(answer=_NO_INFO, citations=[])
    answer = llm.generate(build_prompt(req.query, chunks))
    return QueryResponse(answer=answer, citations=citations_from_chunks(chunks))


@app.post("/query/stream")
def query_stream(req: QueryRequest, retriever=Depends(get_retriever), llm=Depends(get_llm)):
    chunks = retriever.retrieve(req.query, req.top_k)
    citations = citations_from_chunks(chunks)
    prompt = build_prompt(req.query, chunks) if chunks else ""

    def event_stream():
        if not chunks:
            yield sse_frame(_NO_INFO)
        else:
            for token in llm.stream(prompt):
                yield sse_frame(token)
        yield sse_frame(json.dumps([c.model_dump() for c in citations]), event="citations")
        yield sse_frame("[DONE]")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _run_interaction(drug_a: str, drug_b: str, retriever, llm) -> InteractionResponse:
    query = f"{drug_a} with {drug_b} interaction"
    chunks = retriever.retrieve(query, 5)
    answer = llm.generate(build_prompt(query, chunks)) if chunks else ""
    return build_interaction_response(chunks, answer)


@app.post("/interactions/check", response_model=InteractionResponse)
def check_interaction(
    req: EnhancedInteractionRequest, retriever=Depends(get_retriever), llm=Depends(get_llm)
) -> InteractionResponse:
    return _run_interaction(req.drugA, req.drugB, retriever, llm)


@app.post("/interactions/check-enhanced", response_model=InteractionResponse)
def check_interaction_enhanced(
    req: EnhancedInteractionRequest, retriever=Depends(get_retriever), llm=Depends(get_llm)
) -> InteractionResponse:
    return _run_interaction(req.drugA, req.drugB, retriever, llm)


@app.get("/drugs")
def list_drugs() -> dict:
    docs = load_corpus(get_settings().corpus_dir)
    names = sorted({n for d in docs for n in d.all_drug_names()})
    return {"drugs": [{"name": n} for n in names], "count": len(names)}


@app.get("/drugs/{drug_name}")
def get_drug(drug_name: str) -> dict:
    key = drug_name.lower().strip()
    docs = load_corpus(get_settings().corpus_dir)
    for d in docs:
        if key in d.all_drug_names():
            return {"name": key, "interactions": [d.id for d in docs if key in d.all_drug_names()]}
    raise HTTPException(status_code=404, detail="Drug not found")
