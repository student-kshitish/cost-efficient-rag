from sentence_transformers import SentenceTransformer


class Embedder:
    """Generate normalized vector embeddings for text."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name

        if device is None:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = SentenceTransformer(
                model_name,
                device=device,
            )

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ):
        """Convert multiple texts into normalized vectors."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Texts must not contain empty strings")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > batch_size,
        )

        return embeddings

    def embed_query(self, query: str):
        """Convert a single search query into a vector."""
        if not query.strip():
            raise ValueError("Query must not be empty")

        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.model.get_embedding_dimension()
