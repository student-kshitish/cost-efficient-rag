import json
from pathlib import Path

INPUT = Path("evaluation/results/candidates.jsonl")
OUTPUT = Path("evaluation/results/relevance_review.md")

records = [
    json.loads(line)
    for line in INPUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

with OUTPUT.open("w", encoding="utf-8") as f:
    f.write("# Retrieval Relevance Review\n\n")
    f.write("Mark each candidate relevant only if its text contains evidence ")
    f.write("that helps answer the question.\n\n")

    for record in records:
        f.write(f"## {record['question_id']}: {record['question']}\n\n")

        for item in record["ranked_candidates"]:
            f.write(f"### Rank {item['rank']}\n")
            f.write(f"- Chunk ID: `{item['chunk_id']}`\n")
            f.write(f"- Chunk index: {item['chunk_index']}\n")
            f.write(f"- Distance: {item['distance']:.4f}\n")
            f.write("- Relevant: `[ ] Yes` / `[ ] No`\n\n")
            f.write("**Evidence:**\n\n")
            f.write(item["text"].strip() + "\n\n")
            f.write("---\n\n")

print(f"Created {OUTPUT}")
print(f"Questions reviewed: {len(records)}")
print(f"Candidate chunks: {sum(len(r['ranked_candidates']) for r in records)}")
