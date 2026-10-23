import json
import statistics
import time
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.embeddings.embedder import Embedder
from app.storage.vector_store import VectorStore
from app.retrieval.search import RetrievalService

questions_path = Path("evaluation/questions.jsonl")

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
retriever = RetrievalService(embedder=embedder, store=store, top_k=settings.top_k)

latencies = []
result_counts = []

for item in questions:
    start = time.perf_counter()
    results = retriever.retrieve(item["question"], top_k=settings.top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    latencies.append(elapsed_ms)
    result_counts.append(len(results))

    print(
        f"{item['question_id']}: "
        f"latency={elapsed_ms:.2f} ms, "
        f"results={len(results)}"
    )

def percentile(values, p):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction

summary = {
    "questions": len(questions),
    "top_k": settings.top_k,
    "mean_latency_ms": statistics.mean(latencies),
    "p50_latency_ms": percentile(latencies, 50),
    "p95_latency_ms": percentile(latencies, 95),
    "mean_results_returned": statistics.mean(result_counts),
}

output = Path("evaluation/results/retrieval_benchmark.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\nSummary:")
print(json.dumps(summary, indent=2))
print(f"\nSaved to {output}")
