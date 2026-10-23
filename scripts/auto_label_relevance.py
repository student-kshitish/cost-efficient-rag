import argparse
import json
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "evaluation/questions.jsonl"
CANDIDATES_FILE = ROOT / "evaluation/results/candidates.jsonl"
OUTPUT_FILE = ROOT / "evaluation/results/auto_relevance_labels.jsonl"
AUTO_QUESTIONS_FILE = ROOT / "evaluation/questions.auto_labeled.jsonl"

MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def call_ollama(question, candidates):
    passages = []

    for item in candidates:
        text = item.get("text", "")
        passages.append({
            "rank": item.get("rank"),
            "chunk_id": item.get("chunk_id"),
            "text": text[:1800],
        })

    prompt = f"""
You are evaluating information retrieval for a RAG system.

Question:
{question}

Retrieved passages:
{json.dumps(passages, ensure_ascii=False)}

For each passage, decide whether it contains information that directly
answers the question or materially helps answer it.

Rules:
- Topical similarity alone is not enough.
- Do not assume information absent from the passage.
- Mark a passage relevant if it contains useful evidence, even if it
  does not contain the complete answer.
- Do not mark a passage relevant merely because it uses similar words.
- Judge each passage independently.

Return ONLY valid JSON in this format:
{{
  "judgments": [
    {{"rank": 1, "relevant": true, "reason": "short explanation"}}
  ]
}}

Include one judgment for every supplied passage.
"""

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    }).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return json.loads(result["response"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up questions.jsonl and apply machine-generated labels",
    )
    args = parser.parse_args()

    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"Missing file: {QUESTIONS_FILE}")
    if not CANDIDATES_FILE.exists():
        raise SystemExit(
            f"Missing {CANDIDATES_FILE}. Run the candidate collection script first."
        )

    questions = read_jsonl(QUESTIONS_FILE)
    candidate_records = read_jsonl(CANDIDATES_FILE)

    candidate_map = {
        row["question_id"]: row.get("ranked_candidates", [])
        for row in candidate_records
    }

    original_questions = {
        row["question_id"]: row for row in questions
    }

    output_rows = []
    updated_questions = []
    errors = 0

    for index, question_record in enumerate(questions, start=1):
        qid = question_record["question_id"]
        question = question_record["question"]
        candidates = candidate_map.get(qid, [])

        print(f"\n[{index}/{len(questions)}] Judging {qid}...")

        if not candidates:
            print("  WARNING: no retrieved candidates; skipping.")
            errors += 1
            continue

        try:
            result = call_ollama(question, candidates)
            judgments = result.get("judgments", [])

            # If the model omits judgments, retry each passage separately.
            valid_ranks = {
                int(item["rank"])
                for item in candidates
                if item.get("rank") is not None
            }
            judged_ranks = {
                int(j["rank"])
                for j in judgments
                if str(j.get("rank", "")).isdigit()
                and int(j["rank"]) in valid_ranks
            }

            if judged_ranks != valid_ranks:
                print("  Incomplete batch response; retrying passages individually...")
                recovered = []

                for candidate in candidates:
                    expected_rank = int(candidate["rank"])
                    single_result = call_ollama(question, [candidate])
                    single_judgments = single_result.get("judgments", [])

                    if not single_judgments:
                        raise ValueError(
                            f"Model returned no judgment for rank {expected_rank}"
                        )

                    # One passage was sent, so normalize its rank to the
                    # candidate's actual retrieval rank.
                    judgment = dict(single_judgments[0])
                    judgment["rank"] = expected_rank

                    if not isinstance(judgment.get("relevant"), bool):
                        raise ValueError(
                            f"Invalid relevance value for rank {expected_rank}"
                        )

                    recovered.append(judgment)

                judgments = recovered

            valid_ranks = {
                int(item["rank"])
                for item in candidates
                if item.get("rank") is not None
            }

            relevant_ranks = {
                int(j["rank"])
                for j in judgments
                if j.get("relevant") is True
                and str(j.get("rank", "")).isdigit()
                and int(j["rank"]) in valid_ranks
            }

            judged_ranks = {
                int(j["rank"])
                for j in judgments
                if str(j.get("rank", "")).isdigit()
                and int(j["rank"]) in valid_ranks
            }

            # Do not silently treat missing judgments as irrelevant.
            if judged_ranks != valid_ranks:
                raise ValueError(
                    f"Incomplete judgments: expected ranks {sorted(valid_ranks)}, "
                    f"received {sorted(judged_ranks)}"
                )

            relevant_ids = [
                item["chunk_id"]
                for item in candidates
                if int(item["rank"]) in relevant_ranks
            ]

            reasons_by_rank = {
                int(j["rank"]): j.get("reason", "")
                for j in judgments
                if str(j.get("rank", "")).isdigit()
            }

            output_rows.append({
                "question_id": qid,
                "question": question,
                "relevant_chunk_ids": relevant_ids,
                "label_source": "automatic_ollama_judgment",
                "model": MODEL,
                "judgments": [
                    {
                        "rank": int(item["rank"]),
                        "chunk_id": item["chunk_id"],
                        "relevant": int(item["rank"]) in relevant_ranks,
                        "reason": reasons_by_rank.get(int(item["rank"]), ""),
                    }
                    for item in candidates
                ],
            })

            updated = dict(question_record)
            updated["relevant_chunk_ids"] = relevant_ids
            updated_questions.append(updated)

            print(f"  Proposed relevant passages: {len(relevant_ids)}")

        except Exception as exc:
            errors += 1
            print(f"  ERROR: {exc}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved machine judgments: {OUTPUT_FILE}")
    print(f"Questions successfully judged: {len(output_rows)}/{len(questions)}")
    print(f"Questions with errors: {errors}")

    if errors:
        print("Fix errors and rerun before applying labels.")
        return

    with AUTO_QUESTIONS_FILE.open("w", encoding="utf-8") as f:
        for row in updated_questions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved labeled dataset: {AUTO_QUESTIONS_FILE}")

    if args.apply:
        backup = QUESTIONS_FILE.with_name(
            f"questions.backup.{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        )
        shutil.copy2(QUESTIONS_FILE, backup)
        shutil.copy2(AUTO_QUESTIONS_FILE, QUESTIONS_FILE)
        print(f"\nBackup created: {backup}")
        print("Machine-generated labels applied to questions.jsonl.")
    else:
        print("\nOriginal questions.jsonl was NOT changed.")
        print("Review the labels, then rerun with --apply to use them.")


if __name__ == "__main__":
    main()
