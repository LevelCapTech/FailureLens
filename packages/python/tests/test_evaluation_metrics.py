"""Tests for the fault-localization ranking metrics."""

from __future__ import annotations

import math

import pytest

from failurelens.evaluation import (
    RankingCase,
    evaluate,
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    top_k_accuracy,
)


def test_reciprocal_rank_first_hit_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit_or_no_relevant() -> None:
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank(["a", "b"], set()) == 0.0


def test_reciprocal_rank_uses_first_relevant() -> None:
    # Two relevant items; the metric rewards the earliest one (rank 2).
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


def test_hit_at_k() -> None:
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) is True
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) is False
    assert hit_at_k(["a", "b", "c"], set(), k=3) is False


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == pytest.approx(1.0)
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=1) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        hit_at_k(["a"], {"a"}, k=0)
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=-1)


def test_mean_reciprocal_rank_and_top_k() -> None:
    cases = [
        RankingCase(predicted=["a", "b"], relevant={"a"}),  # RR = 1.0
        RankingCase(predicted=["x", "y"], relevant={"y"}),  # RR = 0.5
        RankingCase(predicted=["p", "q"], relevant={"z"}),  # RR = 0.0
    ]
    assert mean_reciprocal_rank(cases) == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert top_k_accuracy(cases, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(cases, k=2) == pytest.approx(2 / 3)


def test_evaluate_bundle() -> None:
    cases = [
        RankingCase(predicted=["a", "b", "c"], relevant={"a"}),
        RankingCase(predicted=["x", "y", "z"], relevant={"z"}),
    ]
    report = evaluate(cases, ks=(1, 3))
    assert report["num_cases"] == 2.0
    assert report["mrr"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert report["top_1_accuracy"] == pytest.approx(0.5)
    assert report["top_3_accuracy"] == pytest.approx(1.0)
    assert report["recall_at_3"] == pytest.approx(1.0)


def test_evaluate_empty() -> None:
    report = evaluate([], ks=(1, 5))
    assert report["num_cases"] == 0.0
    assert report["mrr"] == 0.0
    assert report["top_1_accuracy"] == 0.0
    assert math.isclose(report["recall_at_5"], 0.0)
