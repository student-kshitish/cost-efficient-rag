import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: Extracted document text.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Number of words shared between adjacent chunks.

    Returns:
        A list of chunks containing text and position metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = re.findall(r"\S+", text)

    if not words:
        return []

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        chunks.append({
            "chunk_index": len(chunks),
            "text": " ".join(chunk_words),
            "start_word": start,
            "end_word": end,
        })

        if end == len(words):
            break

    return chunks
