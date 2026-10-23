import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS_FILE = ROOT / "evaluation/questions.jsonl"
CANDIDATES_FILE = ROOT / "evaluation/results/candidates.jsonl"
OUTPUT_FILE = ROOT / "evaluation/results/proposed_relevance_labels.jsonl"

BASE_URL = os.getenv(
    "LLM_BASE_URL", "http://localhost:11434"
).rstrip("/")
MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def question_id(record, index):
    return str(
        record.get("id")
        or record.get("question_id")
        or record.get("qid")
        or f"q{index:03d}"
    )


def question_text(record):
    value = record.get("question") or record.get("query") or record.get("text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing question text: {list(record.keys())}")
    return value.strip()


def get_chunk_id(candidate):
    return (
        candidate.get("chunk_id")
        or candidate.get("id")
        or candidate.get("chunk")
    )


def judge(question, candidate):
    text = candidate.get("text", "")
    if not text.strip():
        return {
            "chunk_id": get_chunk_id(candidate),
            "relevant": None,
            "reason": "Candidate has no text.",
        }

    prompt = f"""
Judge whether the passage is relevant to answering the question.
Relevant means it contains information that directly helps answer the
question. Keyword overlap alone is not enough.

Return JSON only in this form:
{{"relevant": true, "reason": "short explanation"}}

QUESTION:
{question}

PASSAGE:
{text[:4500]}
"""
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "model": MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": "Be a careful and conservative retrieval evaluator.",
                },
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()

    raw = response.json().get("message", {}).get("content", "")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "chunk_id": get_chunk_id(candidate),
            "relevant": None,
            "reason": f"Invalid JSON from judge: {raw[:500]}",
        }

    value = parsed.get("relevant")
    if not isinstance(value, bool):
        return {
            "chunk_id": get_chunk_id(candidate),
            "relevant": None,
            "reason": f"Missing Boolean relevance field: {raw[:500]}",
        }

    return {
        "chunk_id": get_chunk_id(candidate),
        "relevant": value,
        "reason": str(parsed.get("reason", "")),
    }


def main():
    questions = read_jsonl(QUESTIONS_FILE)
    candidate_records = read_jsonl(CANDIDATES_FILE)

    # The candidates file is one aggregate record per question.
    candidate_map = {}
    for record in candidate_records:
        qid = record.get("question_id") or record.get("qid")
        if qid is not None:
            candidate_map[str(qid)] = record

    # Support ordered matching only when there are no usable IDs.
    if not candidate_map and len(candidate_records) == len(questions):
        for i, record in enumerate(candidate_records, 1):
            candidate_map[question_id(questions[i - 1], i)] = record

    previous = {}
    if OUTPUT_FILE.exists():
        for record in read_jsonl(OUTPUT_FILE):
            previous[record["question_id"]] = record

    try:
        requests.get(f"{BASE_URL}/api/tags", timeout=10).raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot connect to Ollama at {BASE_URL}: {exc}")

    output = []
    total_judged = 0

    for i, q in enumerate(questions, 1):
        qid = question_id(q, i)
        text = question_text(q)
        existing = q.get("relevant_chunk_ids", [])

        if qid in previous and previous[qid].get("status") == "proposal_generated":
            print(f"[{i}/{len(questions)}] {qid}: reusing previous proposal")
            output.append(previous[qid])
            continue

        candidate_record = candidate_map.get(qid)

        # Candidate files may have generated IDs that differ from the question
        # IDs. Fall back to matching the question text exactly.
        if candidate_record is None:
            for record in candidate_records:
                if record.get("question", "").strip() == text:
                    candidate_record = record
                    break

        if candidate_record is None:
            print(f"[{i}/{len(questions)}] {qid}: no matching candidate record")
            output.append({
                "question_id": qid,
                "question": text,
                "existing_relevant_chunk_ids": existing,
                "proposed_relevant_chunk_ids": [],
                "judgments": [],
                "status": "needs_candidate_mapping",
                "label_source": "local_llm_proposal_not_ground_truth",
            })
            continue

        candidates = candidate_record.get("ranked_candidates", [])
        if not candidates:
            print(f"[{i}/{len(questions)}] {qid}: ranked_candidates is empty")
            output.append({
                "question_id": qid,
                "question": text,
                "existing_relevant_chunk_ids": existing,
                "proposed_relevant_chunk_ids": [],
                "judgments": [],
                "status": "no_candidates",
                "label_source": "local_llm_proposal_not_ground_truth",
            })
            continue

        print(f"[{i}/{len(questions)}] {qid}: judging {len(candidates)} passages")
        judgments = []

        for candidate in candidates:
            try:
                result = judge(text, candidate)
            except Exception as exc:
                result = {
                    "chunk_id": get_chunk_id(candidate),
                    "relevant": None,
                    "reason": f"Judge error: {type(exc).__name__}: {exc}",
                }

            judgments.append(result)
            total_judged += 1

        proposed_ids = [
            item["chunk_id"]
            for item in judgments
            if item.get("relevant") is True and item.get("chunk_id")
        ]

        record = {
            "question_id": qid,
            "question": text,
            "existing_relevant_chunk_ids": existing,
            "proposed_relevant_chunk_ids": proposed_ids,
            "judgments": judgments,
            "status": (
                "proposal_generated"
                if all(item.get("relevant") is not None for item in judgments)
                else "needs_review"
            ),
            "label_source": "local_llm_proposal_not_ground_truth",
        }
        output.append(record)

        # Save after every question to preserve progress.
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nSummary:")
    print(f"Questions: {len(output)}")
    print(f"Passage judgments attempted: {total_judged}")
    print(f"Questions with proposed labels: {sum(bool(r.get('proposed_relevant_chunk_ids')) for r in output)}")
    print(f"Questions needing candidate mapping: {sum(r.get('status') == 'needs_candidate_mapping' for r in output)}")
    print(f"Questions needing review: {sum(r.get('status') == 'needs_review' for r in output)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
