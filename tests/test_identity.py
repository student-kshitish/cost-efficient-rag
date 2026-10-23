from app.ingestion.identity import (
    document_id_for,
    document_metadata,
    sha256_text,
)


def test_same_path_produces_same_document_id(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()

    document = root / "machine_learning.md"
    document.write_text("Original content", encoding="utf-8")

    first_id = document_id_for(document, root)

    document.write_text("Updated content", encoding="utf-8")

    second_id = document_id_for(document, root)

    assert first_id == second_id


def test_changed_text_produces_different_content_hash():
    first_hash = sha256_text("Original content")
    second_hash = sha256_text("Updated content")

    assert first_hash != second_hash


def test_same_text_produces_same_content_hash():
    first_hash = sha256_text("Machine learning")
    second_hash = sha256_text("Machine learning")

    assert first_hash == second_hash


def test_metadata_contains_expected_fields(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()

    document = root / "machine_learning.md"
    document.write_text("Sample content", encoding="utf-8")

    metadata = document_metadata(
        document,
        root,
        "Sample content",
    )

    assert metadata["document_id"]
    assert metadata["source"] == "machine_learning.md"
    assert metadata["document_type"] == "md"
    assert metadata["content_hash"] == sha256_text("Sample content")
