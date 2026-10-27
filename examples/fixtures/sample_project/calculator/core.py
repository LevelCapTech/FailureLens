"""Core arithmetic helpers for the sample project."""

from collections.abc import Sequence


def divide(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``.

    This is where the ZeroDivisionError surfaces, but the root cause is usually
    the *caller* that passed an invalid denominator.
    """
    return numerator / denominator


def average(values: Sequence[float]) -> float:
    """Return the arithmetic mean of ``values``.

    Bug: this does not guard against an empty sequence, so ``len(values)`` can be
    zero and ``divide`` raises ``ZeroDivisionError``.
    """
    total = sum(values)
    return divide(total, len(values))
