from evaluation.metrics import (
    evaluate_query,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_at_k():
    relevant = {"a", "b"}
    ranked = ["a", "c", "b", "d"]

    assert recall_at_k(relevant, ranked, 2) == 0.5
    assert recall_at_k(relevant, ranked, 3) == 1.0


def test_hit_rate():
    assert hit_rate_at_k({"a"}, ["b", "a"], 2) == 1.0
    assert hit_rate_at_k({"a"}, ["b", "c"], 2) == 0.0


def test_mrr():
    assert mrr({"a"}, ["b", "c", "a"], 3) == 1 / 3
    assert mrr({"a"}, ["b", "c"], 2) == 0.0


def test_ndcg_perfect_ranking():
    assert ndcg_at_k({"a", "b"}, ["a", "b", "c"], 3) == 1.0


def test_precision():
    assert precision_at_k({"a", "b"}, ["a", "c", "d"], 3) == 1 / 3


def test_evaluate_query():
    result = evaluate_query({"a"}, ["a", "b"], k=2)

    assert result["recall@2"] == 1.0
    assert result["hit_rate@2"] == 1.0
    assert result["mrr@2"] == 1.0
