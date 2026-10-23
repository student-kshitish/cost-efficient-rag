import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "evaluation/questions.jsonl"
CANDIDATES = ROOT / "evaluation/results/candidates.jsonl"
OUTPUT = ROOT / "evaluation/results/relevance_labels.csv"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


questions = read_jsonl(QUESTIONS)
candidate_records = read_jsonl(CANDIDATES)

candidate_map = {
    str(record["question_id"]): record
    for record in candidate_records
    if record.get("question_id") is not None
}

fields = [
    "question_id",
    "question",
    "rank",
    "chunk_id",
    "distance",
    "relevant",
    "candidate_text",
]

rows = []

for index, question_record in enumerate(questions, start=1):
    qid = str(
        question_record.get("id")
        or question_record.get("question_id")
        or question_record.get("qid")
        or f"q{index:03d}"
    )

    question = question_record.get("question") or question_record.get("query", "")
    candidate_record = candidate_map.get(qid)

    if candidate_record is None:
        candidate_record = next(
            (
                record for record in candidate_records
                if record.get("question", "").strip() == question.strip()
            ),
            None,
        )

    if candidate_record is None:
        print(f"WARNING: No candidate record found for {qid}")
        continue

    for candidate in candidate_record.get("ranked_candidates", []):
        chunk_id = candidate.get("chunk_id")

        # Preserve an existing verified positive label when available.
        existing_ids = question_record.get("relevant_chunk_ids", [])
        label = 1 if chunk_id in existing_ids else ""

        rows.append({
            "question_id": qid,
            "question": question,
            "rank": candidate.get("rank", ""),
            "chunk_id": chunk_id or "",
            "distance": candidate.get("distance", ""),
            "relevant": label,
            "candidate_text": candidate.get("text", ""),
        })

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Questions: {len(questions)}")
print(f"Candidate passages exported: {len(rows)}")
print(f"CSV created: {OUTPUT}")
print("\nEnter 1 for relevant passages and 0 for irrelevant passages.")
print("Leave a cell blank if you have not reviewed that passage.")
