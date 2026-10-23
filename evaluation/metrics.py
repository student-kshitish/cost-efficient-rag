from math import log2


def recall_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Fraction of all known relevant chunks found in the top k."""
    if not relevant_ids:
        return 0.0

    retrieved = set(ranked_ids[:k])
    return len(relevant_ids & retrieved) / len(relevant_ids)


def hit_rate_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """1 if any relevant chunk appears in the top k; otherwise 0."""
    return float(bool(relevant_ids & set(ranked_ids[:k])))


def reciprocal_rank(relevant_ids: set[str], ranked_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mrr(relevant_ids: set[str], ranked_ids: list[str], k: int | None = None) -> float:
    """MRR; optionally limit ranking to the top k."""
    if k is not None:
        ranked_ids = ranked_ids[:k]

    return reciprocal_rank(relevant_ids, ranked_ids)


def ndcg_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Binary-relevance normalized discounted cumulative gain."""
    ranked_ids = ranked_ids[:k]

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids, start=1)
        if chunk_id in relevant_ids
    )

    ideal_relevant_count = min(len(relevant_ids), k)

    if ideal_relevant_count == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / ideal_dcg


def precision_at_k(relevant_ids: set[str], ranked_ids: list[str], k: int) -> float:
    """Fraction of the returned top-k results that are relevant."""
    retrieved = ranked_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(chunk_id in relevant_ids for chunk_id in retrieved)
    return relevant_count / len(retrieved)


def evaluate_query(
    relevant_ids: set[str],
    ranked_ids: list[str],
    k: int = 5,
) -> dict[str, float]:
    """Calculate retrieval metrics for one labeled question."""
    return {
        f"recall@{k}": recall_at_k(relevant_ids, ranked_ids, k),
        f"hit_rate@{k}": hit_rate_at_k(relevant_ids, ranked_ids, k),
        f"mrr@{k}": mrr(relevant_ids, ranked_ids, k),
        f"ndcg@{k}": ndcg_at_k(relevant_ids, ranked_ids, k),
        f"precision@{k}": precision_at_k(relevant_ids, ranked_ids, k),
    }
