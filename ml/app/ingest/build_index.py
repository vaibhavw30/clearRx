from __future__ import annotations

from app.config import get_settings
from app.rag.chunking import Chunker, FixedSizeChunker
from app.rag.corpus import load_corpus
from app.rag.embeddings import BGEEmbedder, Embedder
from app.rag.models import Monograph
from app.rag.vectorstore import PineconeStore, Record


def build_records(
    docs: list[Monograph], chunker: Chunker, embedder: Embedder
) -> list[Record]:
    chunks = [c for doc in docs for c in chunker.chunk(doc)]
    if not chunks:
        return []
    vectors = embedder.embed([c.text for c in chunks])
    records: list[Record] = []
    for c, vec in zip(chunks, vectors):
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
                metadata=metadata,
            )
        )
    return records


def main() -> None:
    settings = get_settings()
    docs = load_corpus(settings.corpus_dir)
    embedder = BGEEmbedder(settings.embedding_model, settings.embedding_dim)
    store = PineconeStore(settings.pinecone_api_key, settings.pinecone_index)
    store.ensure_index(
        settings.embedding_dim, settings.pinecone_metric,
        settings.pinecone_cloud, settings.pinecone_region,
    )
    records = build_records(docs, FixedSizeChunker(chunk_size=512, overlap=0), embedder)
    store.upsert(records, namespace=settings.pinecone_namespace)
    print(f"upserted {len(records)} chunks from {len(docs)} monographs "
          f"to index '{settings.pinecone_index}' namespace '{settings.pinecone_namespace}'")


if __name__ == "__main__":
    main()
