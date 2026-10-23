import numpy as np
import pytest

from app.embeddings.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embedding_dimension(embedder):
    assert embedder.dimension == 384


def test_embedding_shape(embedder):
    vectors = embedder.embed_texts([
        "Machine learning learns patterns from data.",
        "Vector databases store numerical representations.",
    ])

    assert vectors.shape == (2, 384)


def test_embeddings_are_normalized(embedder):
    vectors = embedder.embed_texts([
        "Machine learning",
        "Vector databases",
    ])

    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_list_returns_empty_list(embedder):
    assert embedder.embed_texts([]) == []


def test_empty_query_is_rejected(embedder):
    with pytest.raises(ValueError):
        embedder.embed_query("   ")


def test_invalid_batch_size_is_rejected(embedder):
    with pytest.raises(ValueError):
        embedder.embed_texts(["Machine learning"], batch_size=0)
