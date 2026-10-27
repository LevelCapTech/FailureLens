"""Ranking metrics for fault-localization evaluation.

These are small, dependency-free, deterministic functions that operate on
*already-ranked* predictions and a set of ground-truth relevant identifiers.
Identifiers are opaque strings chosen by the caller, for example a file path
(``"pkg/mod.py"``) for file-level evaluation or ``"pkg/mod.py::func"`` for
function-level evaluation.

Nothing here fabricates data: the functions only summarise predictions you
supply. See ``evaluation/README.md`` for how the benchmark harness feeds real
diagnoses from :mod:`failurelens` into these metrics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "RankingCase",
    "evaluate",
    "hit_at_k",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
    "top_k_accuracy",
]


@dataclass(frozen=True)
class RankingCase:
    """A single evaluation case.

    Attributes:
        predicted: Ranked identifiers, best first (rank 1 is ``predicted[0]``).
        relevant: Ground-truth identifiers that count as correct.
        case_id: Optional human-readable identifier for reporting.
    """

    predicted: Sequence[str]
    relevant: frozenset[str] = field(default_factory=frozenset)
    case_id: str | None = None

    def __post_init__(self) -> None:
        # Normalise ``relevant`` so callers may pass any iterable.
        if not isinstance(self.relevant, frozenset):
            object.__setattr__(self, "relevant", frozenset(self.relevant))


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")


def reciprocal_rank(predicted: Sequence[str], relevant: Iterable[str]) -> float:
    """Return ``1 / rank`` of the first relevant prediction, else ``0.0``."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for index, item in enumerate(predicted, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def hit_at_k(predicted: Sequence[str], relevant: Iterable[str], k: int) -> bool:
    """Return ``True`` if any relevant identifier appears in the top ``k``."""
    _check_k(k)
    relevant_set = set(relevant)
    return any(item in relevant_set for item in predicted[:k])


def recall_at_k(predicted: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant identifiers found within the top ``k``.

    Returns ``0.0`` when there are no relevant identifiers (nothing to recall).
    """
    _check_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = relevant_set.intersection(predicted[:k])
    return len(found) / len(relevant_set)


def mean_reciprocal_rank(cases: Iterable[RankingCase]) -> float:
    """Mean reciprocal rank across cases. Returns ``0.0`` for no cases."""
    case_list = list(cases)
    if not case_list:
        return 0.0
    total = sum(reciprocal_rank(c.predicted, c.relevant) for c in case_list)
    return total / len(case_list)


def top_k_accuracy(cases: Iterable[RankingCase], k: int) -> float:
    """Fraction of cases with at least one relevant item in the top ``k``."""
    _check_k(k)
    case_list = list(cases)
    if not case_list:
        return 0.0
    hits = sum(1 for c in case_list if hit_at_k(c.predicted, c.relevant, k))
    return hits / len(case_list)


def evaluate(cases: Iterable[RankingCase], ks: Sequence[int] = (1, 5, 10)) -> Mapping[str, float]:
    """Compute a standard metric bundle over ``cases``.

    Returns a mapping with ``num_cases``, ``mrr``, and ``top_{k}_accuracy`` /
    ``recall_at_{k}`` entries for each ``k`` in ``ks``. All values are real
    measurements derived from the supplied predictions.
    """
    case_list = list(cases)
    results: dict[str, float] = {
        "num_cases": float(len(case_list)),
        "mrr": mean_reciprocal_rank(case_list),
    }
    for k in ks:
        results[f"top_{k}_accuracy"] = top_k_accuracy(case_list, k)
        if case_list:
            results[f"recall_at_{k}"] = sum(
                recall_at_k(c.predicted, c.relevant, k) for c in case_list
            ) / len(case_list)
        else:
            results[f"recall_at_{k}"] = 0.0
    return results
