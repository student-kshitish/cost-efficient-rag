import json
from pathlib import Path

from app.config import settings
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.retrieval.search import RetrievalService


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "candidates.jsonl"


def main():
    questions = [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print("Loading embedding model...")
    embedder = Embedder(model_name=settings.embedding_model)

    print("Opening vector database...")
    store = VectorStore(
        db_path=settings.vector_db_path,
        dimension=embedder.dimension,
    )

    retriever = RetrievalService(
        embedder=embedder,
        store=store,
        top_k=5,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for index, item in enumerate(questions, start=1):
            question_id = item["question_id"]
            question = item["question"]

            print(f"\n[{index}/{len(questions)}] {question_id}: {question}")

            results = retriever.retrieve(question=question, top_k=5)

            candidates = []
            for rank, result in enumerate(results, start=1):
                candidate = {
                    "rank": rank,
                    "chunk_id": result["chunk_id"],
                    "source": result["source"],
                    "chunk_index": result["chunk_index"],
                    "distance": result["distance"],
                    "text": result["text"],
                }
                candidates.append(candidate)

                print(
                    f"\n  Rank {rank} | chunk={candidate['chunk_id']} "
                    f"| distance={candidate['distance']:.4f}"
                )
                print(f"  {candidate['text'][:350].replace(chr(10), ' ')}")

            record = {
                "question_id": question_id,
                "question": question,
                "ranked_candidates": candidates,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved candidate evidence to: {OUTPUT_PATH}")
    print("Review the evidence before labeling relevant chunks.")


if __name__ == "__main__":
    main()
