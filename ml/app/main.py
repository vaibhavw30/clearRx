from __future__ import annotations

import json

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from app.answer import citations_from_chunks, sse_frame
from app.deps import get_llm, get_retriever
from app.rag.pipeline import build_prompt
from app.schemas import HealthResponse, QueryRequest, QueryResponse

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
