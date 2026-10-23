import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
BASE_URL = os.getenv(
    "LLM_BASE_URL", "http://localhost:11434"
).rstrip("/")

questions_path = ROOT / "evaluation/questions.jsonl"
candidates_path = ROOT / "evaluation/results/candidates.jsonl"

with questions_path.open(encoding="utf-8") as f:
    question = json.loads(next(line for line in f if line.strip()))

with candidates_path.open(encoding="utf-8") as f:
    candidate_record = json.loads(next(line for line in f if line.strip()))

question_text = question.get("question") or question.get("query")
if not question_text:
    raise SystemExit(f"Cannot find question text: {list(question.keys())}")

ranked = candidate_record.get("ranked_candidates", [])
if not ranked:
    raise SystemExit(
        f"No ranked candidates. Candidate fields: {list(candidate_record.keys())}"
    )

candidate = ranked[0]
passage_text = candidate.get("text", "")
chunk_id = candidate.get("chunk_id")

if not passage_text:
    raise SystemExit(
        f"Cannot find passage text. Fields: {list(candidate.keys())}"
    )

prompt = f"""
Determine whether this passage is relevant to answering the question.
A passage is relevant if it contains information that directly helps answer
the question. Keyword overlap alone is insufficient.

Return JSON only:
{{"relevant": true, "reason": "brief explanation"}}

Question:
{question_text}

Passage:
{passage_text[:4000]}
"""

response = requests.post(
    f"{BASE_URL}/api/chat",
    json={
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0},
    },
    timeout=180,
)
response.raise_for_status()

raw = response.json().get("message", {}).get("content", "")

print("Question:", question_text)
print("Candidate chunk ID:", chunk_id)
print("Candidate text preview:", passage_text[:300])
print("\nRaw judge output:")
print(raw)

try:
    parsed = json.loads(raw)
    print("\nParsed JSON:")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as exc:
    print("\nInvalid JSON returned by judge:", exc)
