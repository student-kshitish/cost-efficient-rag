import pytest

from app.ingestion.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_one_chunk():
    chunks = chunk_text(
    "Machine learning uses data.",
    chunk_size=10,
    chunk_overlap=2,
)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Machine learning uses data."
    assert chunks[0]["chunk_index"] == 0


def test_long_text_is_split_into_chunks():
    text = " ".join(f"word{i}" for i in range(1200))

    chunks = chunk_text(
        text,
        chunk_size=500,
        chunk_overlap=100,
    )

    assert len(chunks) == 3
    assert len(chunks[0]["text"].split()) == 500
    assert len(chunks[1]["text"].split()) == 500
    assert len(chunks[2]["text"].split()) == 400


def test_adjacent_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(20))

    chunks = chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=3,
    )

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()

    assert first_words[-3:] == second_words[:3]


def test_chunk_positions_are_recorded():
    text = "one two three four five six"

    chunks = chunk_text(
        text,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert chunks[0]["start_word"] == 0
    assert chunks[0]["end_word"] == 4
    assert chunks[1]["start_word"] == 3


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_settings_raise_error(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text(
            "some sample text",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
