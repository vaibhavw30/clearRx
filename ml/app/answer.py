from __future__ import annotations

from typing import Optional

from app.rag.models import Chunk
from app.schemas import Citation, InteractionResponse

_FALLBACK_RECOMMENDATION = (
    "Consult a pharmacist or physician before combining these medications."
)

# The corpus severity vocabulary (high/moderate/low) is mapped to the vocabulary
# the dashboard + Express consume (severe/moderate/mild); unknown/missing stays "unknown".
_SEVERITY_MAP = {"high": "severe", "moderate": "moderate", "low": "mild"}


def citations_from_chunks(chunks: list[Chunk]) -> list[Citation]:
    cites: list[Citation] = []
    seen: set = set()
    for c in chunks:
        if c.source_doc_id in seen:
            continue
        seen.add(c.source_doc_id)
        cites.append(
            Citation(
                source_doc_id=c.source_doc_id,
                section=c.section or None,
                url=c.metadata.get("source_url"),
            )
        )
    return cites


def sse_frame(data: str, event: Optional[str] = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


def build_interaction_response(chunks: list[Chunk], answer: str) -> InteractionResponse:
    if not chunks:
        return InteractionResponse(
            severity="unknown",
            description="No interaction information available in the corpus.",
            recommendation=_FALLBACK_RECOMMENDATION,
            sources=[],
            confidence=0.0,
            method="rag",
        )
    severity = _SEVERITY_MAP.get(chunks[0].metadata.get("severity", "unknown"), "unknown")
    management = next((c.text for c in chunks if c.section == "management"), None)
    cites = citations_from_chunks(chunks)
    sources = [c.url or c.source_doc_id for c in cites]
    return InteractionResponse(
        severity=severity,
        description=answer,
        recommendation=management or _FALLBACK_RECOMMENDATION,
        sources=sources,
        confidence=0.85,
        method="rag",
    )
