"""The six fixed v0.6 reranker candidates (evaluator-only; scikit-learn lives in the experiment env).

The candidate set, hyperparameters and the negative sampling are fixed in
``docs/releases/v0.6/selection-rule.md`` before any result and are not tuned. scikit-learn
is imported lazily so the library and the core test environment never need it.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_SEED_BASE = 20261106
TOP_BM25 = 100
RANDOM_NEGATIVES = 100
COL_IN_STACK = 1
COL_MIN_HOPS = 7
COL_BM25 = 9


@dataclass(frozen=True)
class RerankerConfig:
    key: str
    family: str  # logistic | pairwise | boosting
    groups: tuple[str, ...]  # ("G1",) or ("G1", "G2")
    c: float | None = None

    @property
    def uses_embedding(self) -> bool:
        return "G2" in self.groups

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "feature_groups": list(self.groups),
            "c": self.c,
        }


CONFIGS: tuple[RerankerConfig, ...] = (
    RerankerConfig("A", "logistic", ("G1",), 1.0),
    RerankerConfig("B", "logistic", ("G1",), 0.1),
    RerankerConfig("C", "pairwise", ("G1",), 1.0),
    RerankerConfig("D", "boosting", ("G1",)),
    RerankerConfig("E", "boosting", ("G1", "G2")),
    RerankerConfig("F", "logistic", ("G1", "G2"), 1.0),
)


def design_matrix(config: RerankerConfig, g1: np.ndarray, g2: np.ndarray | None) -> np.ndarray:
    if config.uses_embedding:
        if g2 is None:
            raise ValueError("this candidate needs the embedding feature group")
        return np.hstack([g1, g2])
    return g1


@dataclass(frozen=True)
class TrainCase:
    """One training case: the G1 matrix, optional G2 matrix and the positive indices."""

    g1: np.ndarray
    g2: np.ndarray | None
    positives: tuple[int, ...]
    ids: tuple[str, ...]
    index: int  # stable case number for the sampling seed


def training_subset(case: TrainCase) -> np.ndarray:
    """Positives, stack/graph neighbourhood, top BM25 and a seeded random sample of the rest."""
    size = case.g1.shape[0]
    chosen: set[int] = set(case.positives)
    near = np.flatnonzero((case.g1[:, COL_IN_STACK] > 0) | (case.g1[:, COL_MIN_HOPS] <= 2))
    chosen.update(int(i) for i in near)
    order = sorted(range(size), key=lambda i: (-case.g1[i, COL_BM25], case.ids[i]))
    chosen.update(order[:TOP_BM25])
    remainder = np.array(sorted(set(range(size)) - chosen), dtype=np.int64)
    if remainder.size:
        rng = np.random.RandomState(SAMPLE_SEED_BASE + case.index)
        take = min(RANDOM_NEGATIVES, remainder.size)
        chosen.update(int(i) for i in rng.choice(remainder, size=take, replace=False))
    return np.array(sorted(chosen), dtype=np.int64)


class FittedReranker:
    """A fitted candidate. ``score`` returns one number per row; higher ranks first."""

    def __init__(self, config: RerankerConfig, scaler: Any, model: Any, weights: Any) -> None:
        self.config = config
        self._scaler = scaler
        self._model = model
        self._weights = weights

    def score(self, matrix: np.ndarray) -> np.ndarray:
        if self.config.family == "boosting":
            return np.asarray(self._model.decision_function(matrix), dtype=np.float64)
        scaled = self._scaler.transform(matrix)
        if self.config.family == "pairwise":
            return np.asarray(scaled @ self._weights, dtype=np.float64)
        return np.asarray(self._model.decision_function(scaled), dtype=np.float64)

    def coefficients(self) -> list[float] | None:
        """Standardized linear weights (None for the boosted trees)."""
        if self.config.family == "pairwise":
            return [float(v) for v in self._weights]
        if self.config.family == "logistic":
            return [float(v) for v in self._model.coef_[0]]
        return None


def fit_reranker(config: RerankerConfig, cases: Sequence[TrainCase]) -> FittedReranker:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    per_case: list[tuple[np.ndarray, np.ndarray]] = []
    for case in cases:
        if not case.positives:
            continue
        subset = training_subset(case)
        matrix = design_matrix(config, case.g1, case.g2)[subset]
        label = np.isin(subset, np.array(case.positives)).astype(np.int64)
        blocks.append(matrix)
        labels.append(label)
        per_case.append((matrix, label))
    if not blocks:
        raise ValueError("no training case has its positive function in the candidate pool")
    stacked = np.vstack(blocks)
    target = np.concatenate(labels)
    scaler = StandardScaler().fit(stacked)
    if config.family == "logistic":
        assert config.c is not None
        model = LogisticRegression(C=config.c, max_iter=2000).fit(scaler.transform(stacked), target)
        return FittedReranker(config, scaler, model, None)
    if config.family == "pairwise":
        assert config.c is not None
        diffs: list[np.ndarray] = []
        for matrix, label in per_case:
            scaled = scaler.transform(matrix)
            positive, negative = scaled[label == 1], scaled[label == 0]
            if positive.size == 0 or negative.size == 0:
                continue
            diffs.append((positive[:, None, :] - negative[None, :, :]).reshape(-1, scaled.shape[1]))
        paired = np.vstack(diffs)
        data = np.vstack([paired, -paired])
        signs = np.concatenate(
            [np.ones(len(paired), dtype=np.int64), np.zeros(len(paired), dtype=np.int64)]
        )
        model = LogisticRegression(C=config.c, fit_intercept=False, max_iter=2000).fit(data, signs)
        return FittedReranker(config, scaler, model, model.coef_[0])
    if config.family == "boosting":
        model = HistGradientBoostingClassifier(
            max_depth=3,
            max_leaf_nodes=8,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=0,
            early_stopping=False,
        ).fit(stacked, target)
        return FittedReranker(config, scaler, model, None)
    raise ValueError(f"unknown family {config.family!r}")


def save_artifact(model: FittedReranker, path: Path) -> str:
    """Write a local pickle (owner-only) and return its SHA-256."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(model, protocol=4)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def load_artifact(path: Path, expected_sha256: str) -> FittedReranker:
    """Load a pickle only after its digest matches; never load unverified files."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("artifact digest mismatch; refusing to load")
    loaded = pickle.loads(payload)
    if not isinstance(loaded, FittedReranker):
        raise ValueError("artifact is not a FittedReranker")
    return loaded
