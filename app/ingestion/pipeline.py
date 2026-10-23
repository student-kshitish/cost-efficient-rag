from pathlib import Path

from app.embeddings.embedder import Embedder
from app.ingestion.chunker import chunk_text
from app.ingestion.identity import document_metadata
from app.ingestion.loader import load_document
from app.storage.vector_store import VectorStore


def ingest_file(
    path: str | Path,
    document_root: str | Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> dict:
    """
    Load, chunk, embed, and store a document.

    Unchanged documents are skipped. Changed documents have their
    previous chunks replaced.
    """
    path = Path(path).resolve()
    document_root = Path(document_root).resolve()

    if not path.is_relative_to(document_root):
        raise ValueError("Document must be inside document_root")

    # 1. Extract text.
    loaded = load_document(path)
    text = loaded["text"]

    # 2. Build stable document identity and metadata.
    metadata = document_metadata(
        path=path,
        document_root=document_root,
        extracted_text=text,
    )

    document_id = metadata["document_id"]
    content_hash = metadata["content_hash"]
    source = path.relative_to(document_root).as_posix()

    # 3. Skip unchanged documents.
    stored_hash = store.get_document_hash(document_id)

    if stored_hash == content_hash:
        return {
            "status": "skipped",
            "reason": "unchanged",
            "document_id": document_id,
            "source": source,
            "chunks_added": 0,
        }

    # 4. Chunk the extracted text.
    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(f"No chunks generated for {source}")

    # 5. Embed the chunks before deleting existing records.
    vectors = embedder.embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")

    # 6. Prepare records for storage.
    records = []

    for chunk, vector in zip(chunks, vectors):
        records.append({
            "chunk_id": (
                f"{document_id}:{content_hash[:12]}:"
                f"{chunk['chunk_index']}"
            ),
            "document_id": document_id,
            "source": source,
            "document_type": metadata["document_type"],
            "content_hash": content_hash,
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "vector": vector,
        })

    # 7. Replace old chunks only after preparing the new records.
    if stored_hash is not None:
        store.delete_document(document_id)

    inserted = store.add_chunks(records)

    return {
        "status": "ingested",
        "document_id": document_id,
        "source": source,
        "content_hash": content_hash,
        "chunks_added": inserted,
    }
