from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class Citation(BaseModel):
    source_doc_id: str
    section: Optional[str] = None
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    services: dict = Field(default_factory=dict)


class EnhancedInteractionRequest(BaseModel):
    drugA: str
    drugB: str
    patientContext: dict = Field(default_factory=dict)


class InteractionResponse(BaseModel):
    severity: str
    description: str
    recommendation: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    method: str
