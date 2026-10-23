import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = ROOT / "evaluation/questions.jsonl"
CANDIDATES_FILE = ROOT / "evaluation/results/candidates.jsonl"
OUTPUT_FILE = ROOT / "evaluation/results/relevance_review_auto.md"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


questions = read_jsonl(QUESTIONS_FILE)
candidate_records = read_jsonl(CANDIDATES_FILE)

candidate_map = {
    str(record.get("question_id")): record
    for record in candidate_records
    if record.get("question_id") is not None
}

lines = [
    "# Retrieval Relevance Review",
    "",
    "Review each candidate passage against its question.",
    "",
    "Mark each passage RELEVANT or NOT RELEVANT based on whether it",
    "directly helps answer the question. These labels are not final until",
    "reviewed. Do not label a passage relevant based on keywords alone.",
    "",
]

for index, question in enumerate(questions, start=1):
    qid = str(
        question.get("id")
        or question.get("question_id")
        or question.get("qid")
        or f"q{index:03d}"
    )
    question_text = question.get("question") or question.get("query") or ""
    existing = question.get("relevant_chunk_ids", [])

    record = candidate_map.get(qid)

    if record is None:
        record = next(
            (
                item for item in candidate_records
                if item.get("question", "").strip() == question_text.strip()
            ),
            None,
        )

    lines.extend([
        "",
        "---",
        "",
        f"## {qid}",
        "",
        f"**Question:** {question_text}",
        "",
        f"**Existing verified relevant chunk IDs:** {json.dumps(existing)}",
        "",
    ])

    if not record:
        lines.extend(["**ERROR:** No matching candidate record found.", ""])
        continue

    candidates = record.get("ranked_candidates", [])

    for candidate in candidates:
        chunk_id = candidate.get("chunk_id", "MISSING_CHUNK_ID")
        rank = candidate.get("rank", "?")
        distance = candidate.get("distance", "unknown")
        text = candidate.get("text", "")

        lines.extend([
            f"### Rank {rank} — `{chunk_id}`",
            "",
            f"- Distance: `{distance}`",
            "- Relevance: **[REVIEW: RELEVANT / NOT RELEVANT]**",
            "",
            "<details>",
            "<summary>Read retrieved passage</summary>",
            "",
            text,
            "",
            "</details>",
            "",
        ])

OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

print(f"Questions: {len(questions)}")
print(f"Candidate records: {len(candidate_records)}")
print(f"Review file created: {OUTPUT_FILE}")
print("Open this Markdown file and review the passages.")
