import numpy as np
import pytest

from app.storage.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    return VectorStore(
        db_path=str(tmp_path / "test_lancedb"),
        dimension=3,
    )


def make_chunk(
    chunk_id,
    text,
    vector,
    document_type="pdf",
):
    return {
        "chunk_id": chunk_id,
        "document_id": "document_1",
        "source": "sample.pdf",
        "document_type": document_type,
        "content_hash": "test_hash",
        "chunk_index": 0,
        "text": text,
        "vector": vector,
    }


def test_insert_and_count(store):
    chunks = [
        make_chunk(
            "chunk_1",
            "Machine learning learns patterns.",
            [1.0, 0.0, 0.0],
        ),
        make_chunk(
            "chunk_2",
            "Vector databases store embeddings.",
            [0.0, 1.0, 0.0],
        ),
    ]

    inserted = store.add_chunks(chunks)

    assert inserted == 2
    assert store.count() == 2


def test_search_returns_similar_chunk(store):
    store.add_chunks([
        make_chunk(
            "chunk_1",
            "Machine learning learns patterns.",
            [1.0, 0.0, 0.0],
        ),
        make_chunk(
            "chunk_2",
            "Vector databases store embeddings.",
            [0.0, 1.0, 0.0],
        ),
    ])

    results = store.search(
        [0.99, 0.01, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_1"


def test_metadata_filter(store):
    store.add_chunks([
        make_chunk(
            "chunk_pdf",
            "PDF content",
            [1.0, 0.0, 0.0],
            "pdf",
        ),
        make_chunk(
            "chunk_md",
            "Markdown content",
            [0.99, 0.01, 0.0],
            "md",
        ),
    ])

    results = store.search(
        [1.0, 0.0, 0.0],
        top_k=5,
        document_type="md",
    )

    assert len(results) == 1
    assert results[0]["document_type"] == "md"


def test_delete_document(store):
    store.add_chunks([
        make_chunk(
            "chunk_1",
            "First chunk",
            [1.0, 0.0, 0.0],
        ),
        make_chunk(
            "chunk_2",
            "Second chunk",
            [0.0, 1.0, 0.0],
        ),
    ])

    store.delete_document("document_1")

    assert store.count() == 0


def test_wrong_vector_dimension_is_rejected(store):
    chunk = make_chunk(
        "chunk_1",
        "Test text",
        [1.0, 0.0],
    )

    with pytest.raises(ValueError, match="dimension"):
        store.add_chunks([chunk])


def test_empty_insert(store):
    assert store.add_chunks([]) == 0
    assert store.count() == 0
