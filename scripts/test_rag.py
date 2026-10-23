from pprint import pprint

from app.config import settings
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.retrieval.search import RetrievalService
from app.generation.answer import AnswerService


def main():
    print("Initializing embedding model...")

    embedder = Embedder()

    print("Connecting to vector database...")

    store = VectorStore(
        db_path=settings.vector_db_path,
        dimension=embedder.dimension,
    )

    retriever = RetrievalService(
        embedder=embedder,
        store=store,
        top_k=5,
    )

    answer_service = AnswerService(
    retriever=retriever,
    model=settings.llm_model,
    base_url=settings.llm_base_url,
)

    question = (
        "What is the difference between storing information in an "
        "attractor basin and carrying analog information?"
    )

    print(f"\nQuestion: {question}")
    print("\nGenerating answer...\n")

    result = answer_service.answer_question(question)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result["answer"])

    print("\n" + "=" * 70)
    print("CITATIONS")
    print("=" * 70)
    pprint(result.get("citations", []))

    print("\n" + "=" * 70)
    print("RETRIEVAL SUMMARY")
    print("=" * 70)

    print(f"Number of retrieved chunks: {result.get('retrieved_chunks', 0)}")

    for citation in result.get("citations", []):
        print(f"\nSource: {citation.get('source')}")
        print(f"Chunk index: {citation.get('chunk_index')}")
        print(f"Citation ID: {citation.get('id')}")
        print(f"Cosine distance: {citation.get('distance')}")

    print(f"Number of retrieved chunks: {result.get('retrieved_chunks', 0)}")

    for citation in result.get("citations", []):
        print(f"\nSource: {citation.get('source')}")
        print(f"Chunk index: {citation.get('chunk_index')}")
        print(f"Citation ID: {citation.get('id')}")
        print(f"Cosine distance: {citation.get('distance')}")

    print("\n" + "=" * 70)
    print("METRICS")
    print("=" * 70)
    pprint({
        "latency_ms": result.get("latency_ms"),
        "model": result.get("model"),
        "usage": result.get("usage"),
    })


if __name__ == "__main__":
    main()
