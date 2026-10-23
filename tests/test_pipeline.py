from app.ingestion.pipeline import ingest_file
from app.storage.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic three-dimensional embeddings for tests."""

    def embed_texts(self, texts, batch_size=32):
        return [
            [1.0, 0.0, 0.0]
            for _ in texts
        ]


def make_document(root, content):
    path = root / "sample.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_ingest_document(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()

    path = make_document(
        root,
        "# Machine Learning\n\nMachine learning learns from data.",
    )

    store = VectorStore(
        db_path=str(tmp_path / "db"),
        dimension=3,
    )

    result = ingest_file(
        path=path,
        document_root=root,
        embedder=FakeEmbedder(),
        store=store,
        chunk_size=20,
        chunk_overlap=2,
    )

    assert result["status"] == "ingested"
    assert result["chunks_added"] == 1
    assert store.count() == 1


def test_reingest_unchanged_document_skips_duplicates(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()

    path = make_document(
        root,
        "# Machine Learning\n\nMachine learning learns from data.",
    )

    store = VectorStore(
        db_path=str(tmp_path / "db"),
        dimension=3,
    )

    first = ingest_file(
        path, root, FakeEmbedder(), store,
        chunk_size=20, chunk_overlap=2,
    )

    second = ingest_file(
        path, root, FakeEmbedder(), store,
        chunk_size=20, chunk_overlap=2,
    )

    assert first["status"] == "ingested"
    assert second["status"] == "skipped"
    assert second["reason"] == "unchanged"
    assert store.count() == 1


def test_changed_document_replaces_old_chunks(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()

    path = make_document(
        root,
        "# Machine Learning\n\nOriginal document content.",
    )

    store = VectorStore(
        db_path=str(tmp_path / "db"),
        dimension=3,
    )

    first = ingest_file(
        path, root, FakeEmbedder(), store,
        chunk_size=20, chunk_overlap=2,
    )

    path.write_text(
        "# Machine Learning\n\nUpdated document content.",
        encoding="utf-8",
    )

    second = ingest_file(
        path, root, FakeEmbedder(), store,
        chunk_size=20, chunk_overlap=2,
    )

    assert first["content_hash"] != second["content_hash"]
    assert second["status"] == "ingested"
    assert store.count() == 1


def test_document_outside_root_is_rejected(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()

    outside = tmp_path / "outside.md"
    outside.write_text("Some text", encoding="utf-8")

    store = VectorStore(
        db_path=str(tmp_path / "db"),
        dimension=3,
    )

    try:
        ingest_file(
            outside, root, FakeEmbedder(), store,
        )
    except ValueError as error:
        assert "inside document_root" in str(error)
    else:
        raise AssertionError("Expected ValueError")
