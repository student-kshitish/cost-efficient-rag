import sys
import json
import time
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.retrieval.search import RetrievalService
from app.generation.answer import AnswerService


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower

    return values[lower] * (1 - fraction) + values[upper] * fraction


questions_path = PROJECT_ROOT / "evaluation" / "questions.jsonl"
output_path = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "answer_quality_samples.jsonl"
)

questions = [
    json.loads(line)
    for line in questions_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

embedder = Embedder(model_name=settings.embedding_model)

store = VectorStore(
    db_path=settings.vector_db_path,
    dimension=embedder.dimension,
)

retriever = RetrievalService(
    embedder=embedder,
    store=store,
    top_k=settings.top_k,
)

answer_service = AnswerService(
    retriever=retriever,
    model=settings.llm_model,
    base_url=settings.llm_base_url,
)

latencies = []
prompt_tokens_total = 0
completion_tokens_total = 0
successful_answers = 0
errors = []

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as output:
    for item in questions:
        question_id = item["question_id"]
        question = item["question"]

        print(f"\nEvaluating {question_id}: {question}")

        start = time.perf_counter()

        try:
            result = answer_service.answer_question(
                question,
                top_k=settings.top_k,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            # Retrieve passages for the evaluation record.
            # These passages let us manually verify answer faithfulness.
            chunks = retriever.retrieve(
                question,
                top_k=settings.top_k,
            )

            usage = result.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0

            prompt_tokens_total += prompt_tokens
            completion_tokens_total += completion_tokens

            answer = result.get("answer", "")
            citations = result.get("citations") or []

            record = {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "citations": citations,
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "source": chunk.get("source"),
                        "chunk_index": chunk.get("chunk_index"),
                        "distance": chunk.get("distance"),
                        "text": chunk.get("text"),
                    }
                    for chunk in chunks
                ],
                "latency_ms": round(elapsed_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "citation_present": bool(citations),
                "answer_nonempty": bool(answer.strip()),
            }

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

            successful_answers += 1

            print(f"  Latency: {elapsed_ms:.2f} ms")
            print(f"  Retrieved passages: {len(chunks)}")
            print(f"  Citations: {len(citations)}")
            print(f"  Prompt tokens: {prompt_tokens}")
            print(f"  Completion tokens: {completion_tokens}")
            print(f"  Answer: {answer[:250].replace(chr(10), ' ')}")

        except Exception as exc:
            errors.append({
                "question_id": question_id,
                "error": str(exc),
            })
            print(f"  ERROR: {exc}")

summary = {
    "questions_total": len(questions),
    "successful_answers": successful_answers,
    "failed_answers": len(errors),
    "mean_latency_ms": (
        round(statistics.mean(latencies), 2) if latencies else 0
    ),
    "p50_latency_ms": round(percentile(latencies, 50), 2),
    "p95_latency_ms": round(percentile(latencies, 95), 2),
    "total_prompt_tokens": prompt_tokens_total,
    "total_completion_tokens": completion_tokens_total,
    "errors": errors,
    "samples_file": str(output_path.relative_to(PROJECT_ROOT)),
}

summary_path = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "answer_quality_summary.json"
)

summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("ANSWER EVALUATION SUMMARY")
print("=" * 60)
print(json.dumps(summary, indent=2))
print(f"\nSaved samples to: {output_path}")
print(f"Saved summary to: {summary_path}")
