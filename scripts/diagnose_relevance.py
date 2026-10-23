import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

proposal_file = ROOT / "evaluation/results/proposed_relevance_labels.jsonl"
candidate_file = ROOT / "evaluation/results/candidates.jsonl"
question_file = ROOT / "evaluation/questions.jsonl"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


questions = read_jsonl(question_file)
candidates = read_jsonl(candidate_file)
proposals = read_jsonl(proposal_file)

candidate_map = {
    str(item.get("question_id")): item
    for item in candidates
    if item.get("question_id") is not None
}

proposal_map = {
    str(item.get("question_id")): item
    for item in proposals
}

for index, question in enumerate(questions, start=1):
    qid = str(
        question.get("id")
        or question.get("question_id")
        or question.get("qid")
        or f"q{index:03d}"
    )

    print("\n" + "=" * 80)
    print(f"QUESTION: {qid}")
    print(question.get("question") or question.get("query"))

    proposal = proposal_map.get(qid, {})
    candidate_record = candidate_map.get(qid, {})
    ranked = candidate_record.get("ranked_candidates", [])
    judgments = {
        str(j.get("chunk_id")): j
        for j in proposal.get("judgments", [])
    }

    print("Existing labels:", question.get("relevant_chunk_ids", []))
    print("Proposed labels:", proposal.get("proposed_relevant_chunk_ids", []))
    print("Candidates found:", len(ranked))

    for candidate in ranked[:2]:
        cid = str(candidate.get("chunk_id"))
        judgment = judgments.get(cid, {})

        print(f"\n  Rank: {candidate.get('rank')}")
        print(f"  Chunk ID: {cid}")
        print(f"  Distance: {candidate.get('distance')}")
        print(f"  Judge decision: {judgment.get('relevant')}")
        print(f"  Judge reason: {judgment.get('reason')}")
        print(f"  Passage: {candidate.get('text', '')[:700]}")

    if index >= 3:
        break
