"""Evaluation utilities for FailureLens.

Reusable, dependency-free ranking metrics for fault localization live in
:mod:`failurelens.evaluation.metrics`. The top-level ``evaluation/`` directory
in the repository holds the benchmark *harness* that feeds real diagnoses into
these functions; see ``evaluation/README.md``.
"""

from __future__ import annotations

from failurelens.evaluation.metrics import (
    RankingCase,
    evaluate,
    hit_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    top_k_accuracy,
)

__all__ = [
    "RankingCase",
    "evaluate",
    "hit_at_k",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
    "top_k_accuracy",
]
