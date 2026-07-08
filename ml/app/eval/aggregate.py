from __future__ import annotations


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def distinct_doc_ids(chunks) -> list[str]:
    """Collapse retrieved chunks to distinct source docs in first-seen order.
    Relevance is judged at the document level, so doc-level ranking metrics
    must see each document once, at the rank of its best (first) chunk."""
    ordered: list[str] = []
    seen: set = set()
    for c in chunks:
        if c.source_doc_id not in seen:
            seen.add(c.source_doc_id)
            ordered.append(c.source_doc_id)
    return ordered
