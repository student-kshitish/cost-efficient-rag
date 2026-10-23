from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore


class RetrievalService:
    """Retrieve relevant document chunks using vector similarity."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        document_type: str | None = None,
    ) -> list[dict]:
        """
        Return the most relevant chunks for a question.

        Results contain the chunk text, source metadata, and
        LanceDB's cosine-distance score.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        k = self.top_k if top_k is None else top_k

        if k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Convert the user's question into a vector.
        query_vector = self.embedder.embed_query(question)

        # Search the vector database.
        results = self.store.search(
            query_vector=query_vector,
            top_k=k,
            document_type=document_type,
        )

        # Return a consistent response format.
        retrieved = []

        for result in results:
            retrieved.append({
                "chunk_id": result["chunk_id"],
                "document_id": result["document_id"],
                "source": result["source"],
                "document_type": result["document_type"],
                "chunk_index": result["chunk_index"],
                "text": result["text"],
                "distance": float(result["_distance"]),
            })

        return retrieved
