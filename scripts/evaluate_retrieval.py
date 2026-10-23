import csv
import json
import statistics
from pathlib import Path

from evaluation.metrics import evaluate_query


ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = ROOT / "evaluation" / "questions.jsonl"
CANDIDATES_FILE = ROOT / "evaluation" / "results" / "candidates.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"

K = 5


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in {path}, line {line_number}: {exc}"
            ) from exc

    return records


def main():
    questions = read_jsonl(QUESTIONS_FILE)
    candidate_records = read_jsonl(CANDIDATES_FILE)

    if not questions:
        raise ValueError("The question dataset is empty.")

    questions_by_id = {}

    for question in questions:
        question_id = question.get("question_id")
        relevant_ids = question.get("relevant_chunk_ids")

        if not question_id:
            raise ValueError("A question is missing question_id.")

        if question_id in questions_by_id:
            raise ValueError(f"Duplicate question ID: {question_id}")

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"{question_id} has no verified relevant_chunk_ids. "
                "Label the dataset before evaluating."
            )

        if not all(isinstance(item, str) and item for item in relevant_ids):
            raise ValueError(
                f"{question_id} contains an invalid relevant chunk ID."
            )

        questions_by_id[question_id] = question

    candidates_by_id = {}

    for record in candidate_records:
        question_id = record.get("question_id")

        if not question_id:
            raise ValueError("A candidate record is missing question_id.")

        if question_id in candidates_by_id:
            raise ValueError(
                f"Duplicate candidate record for {question_id}"
            )

        candidates_by_id[question_id] = record

    missing = set(questions_by_id) - set(candidates_by_id)

    if missing:
        raise ValueError(
            "Candidate rankings are missing for: "
            + ", ".join(sorted(missing))
        )

    rows = []

    for question_id, question in questions_by_id.items():
        record = candidates_by_id[question_id]

        ranked_candidates = sorted(
            record.get("ranked_candidates", []),
            key=lambda item: item["rank"],
        )

        ranked_ids = [
            candidate["chunk_id"]
            for candidate in ranked_candidates
        ]

        relevant_ids = set(question["relevant_chunk_ids"])

        metrics = evaluate_query(
            relevant_ids=relevant_ids,
            ranked_ids=ranked_ids,
            k=K,
        )

        row = {
            "question_id": question_id,
            "question": question["question"],
            "relevant_chunk_count": len(relevant_ids),
            "retrieved_chunk_count": len(ranked_ids),
            **metrics,
        }

        rows.append(row)

    metric_names = [
        f"recall@{K}",
        f"hit_rate@{K}",
        f"mrr@{K}",
        f"ndcg@{K}",
        f"precision@{K}",
    ]

    summary = {
        metric: statistics.mean(row[metric] for row in rows)
        for metric in metric_names
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_output = {
        "evaluation": {
            "k": K,
            "question_count": len(rows),
            "metrics": summary,
            "per_question": rows,
        }
    }

    json_path = RESULTS_DIR / "retrieval_metrics.json"
    csv_path = RESULTS_DIR / "retrieval_metrics.csv"

    json_path.write_text(
        json.dumps(json_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluation complete: {len(rows)} questions")
    print(f"Cutoff: k={K}")
    print("\nMean retrieval metrics:")

    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nJSON report: {json_path}")
    print(f"CSV report:  {csv_path}")


if __name__ == "__main__":
    main()
