import pytest

from app.retrieval.search import RetrievalService
from app.storage.vector_store import VectorStore


class FakeEmbedder:
    """Small deterministic embedder for unit tests."""

    def embed_query(self, question):
        if "machine learning" in question.lower():
            return [1.0, 0.0, 0.0]

        if "vector database" in question.lower():
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]


def make_chunk(
    chunk_id,
    text,
    vector,
    document_type="pdf",
):
    return {
        "chunk_id": chunk_id,
        "document_id": f"doc_{chunk_id}",
        "source": f"{chunk_id}.pdf",
        "document_type": document_type,
        "content_hash": f"hash_{chunk_id}",
        "chunk_index": 0,
        "text": text,
        "vector": vector,
    }


@pytest.fixture
def setup_retrieval(tmp_path):
    store = VectorStore(
        db_path=str(tmp_path / "retrieval_db"),
        dimension=3,
    )

    store.add_chunks([
        make_chunk(
            "ml",
            "Machine learning learns patterns from data.",
            [1.0, 0.0, 0.0],
        ),
        make_chunk(
            "vector",
            "A vector database stores numerical vectors.",
            [0.0, 1.0, 0.0],
        ),
        make_chunk(
            "neural",
            "Neural networks contain interconnected neurons.",
            [0.0, 0.0, 1.0],
            document_type="md",
        ),
    ])

    service = RetrievalService(
        embedder=FakeEmbedder(),
        store=store,
        top_k=2,
    )

    return service, store


def test_retrieval_returns_top_k_results(setup_retrieval):
    service, _ = setup_retrieval

    results = service.retrieve("Explain machine learning")

    assert len(results) == 2
    assert results[0]["chunk_id"] == "ml"


def test_results_include_source_and_text(setup_retrieval):
    service, _ = setup_retrieval

    results = service.retrieve("Explain machine learning")

    assert results[0]["source"] == "ml.pdf"
    assert "Machine learning" in results[0]["text"]
    assert "distance" in results[0]


def test_document_type_filter(setup_retrieval):
    service, _ = setup_retrieval

    results = service.retrieve(
        "Explain machine learning",
        top_k=5,
        document_type="md",
    )

    assert len(results) == 1
    assert results[0]["document_type"] == "md"


def test_empty_question_is_rejected(setup_retrieval):
    service, _ = setup_retrieval

    with pytest.raises(ValueError, match="cannot be empty"):
        service.retrieve("   ")


def test_invalid_top_k_is_rejected(setup_retrieval):
    service, _ = setup_retrieval

    with pytest.raises(ValueError, match="top_k"):
        service.retrieve("Explain machine learning", top_k=0)


def test_empty_database_returns_no_results(tmp_path):
    store = VectorStore(
        db_path=str(tmp_path / "empty_db"),
        dimension=3,
    )

    service = RetrievalService(
        embedder=FakeEmbedder(),
        store=store,
    )

    assert service.retrieve("Explain machine learning") == []
