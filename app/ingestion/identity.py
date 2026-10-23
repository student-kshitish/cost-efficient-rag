import hashlib
import re
from pathlib import Path


def sha256_text(text: str) -> str:
    """Return a SHA-256 hash of normalized text."""
    normalized_text = text.replace("\r\n", "\n")

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def document_id_for(
    path: str | Path,
    document_root: str | Path,
) -> str:
    """
    Generate a stable document ID from its relative path.

    The ID remains unchanged when the document's contents change,
    provided the document stays at the same relative path.
    """
    path = Path(path).resolve()
    root = Path(document_root).resolve()

    relative_path = path.relative_to(root).as_posix().lower()

    readable_name = Path(relative_path).stem.lower()
    readable_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        readable_name,
    ).strip("_")

    path_hash = hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()[:10]

    return f"{readable_name}_{path_hash}"


def document_metadata(
    path: str | Path,
    document_root: str | Path,
    extracted_text: str,
) -> dict:
    """Build basic metadata for an extracted document."""
    path = Path(path).resolve()
    root = Path(document_root).resolve()

    return {
        "document_id": document_id_for(path, root),
        "source": path.relative_to(root).as_posix(),
        "document_type": path.suffix.lower().lstrip("."),
        "content_hash": sha256_text(extracted_text),
    }
