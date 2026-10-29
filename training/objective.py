"""Tiny InfoNCE objective and toy encoder used by the offline smoke pipeline."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class ToyEncoder:
    """A deterministic, tiny linear encoder for offline CPU smoke training."""

    def __init__(self, dim: int = 8, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        rng = random.Random(seed)
        self.weights = [
            [rng.uniform(-0.15, 0.15) for _ in range(self.dim)] for _ in range(self.dim)
        ]

    def _features(self, text: str) -> list[float]:
        features = [0.0 for _ in range(self.dim)]
        for token in _TOKEN_RE.findall(text.lower()):
            index = sum(ord(char) for char in token) % self.dim
            features[index] += 1.0
        return features

    def encode(self, text: str) -> list[float]:
        features = self._features(text)
        return [
            sum(
                self.weights[row_index][column_index] * features[column_index]
                for column_index in range(self.dim)
            )
            for row_index in range(self.dim)
        ]

    def apply_gradients(self, gradients: Iterable[Iterable[float]], lr: float) -> None:
        for row_index, row in enumerate(gradients):
            for column_index, value in enumerate(row):
                self.weights[row_index][column_index] -= lr * value


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class InfoNCEObjective:
    """Small InfoNCE loss with explicit backward pass for the toy model."""

    def loss_and_gradients(
        self,
        encoder: ToyEncoder,
        query: str,
        positive: str,
        negative: str,
    ) -> tuple[float, list[list[float]]]:
        q = encoder.encode(query)
        p = encoder.encode(positive)
        n = encoder.encode(negative)
        sim_pos = _dot(q, p)
        sim_neg = _dot(q, n)
        exp_pos = math.exp(sim_pos)
        exp_neg = math.exp(sim_neg)
        denom = exp_pos + exp_neg
        loss = math.log(denom) - sim_pos

        prob_neg = exp_neg / denom
        grad_q = [
            prob_neg * (neg_value - pos_value) for pos_value, neg_value in zip(p, n, strict=True)
        ]
        grad_p = [-prob_neg * value for value in q]
        grad_n = [prob_neg * value for value in q]

        features = [
            encoder._features(query),
            encoder._features(positive),
            encoder._features(negative),
        ]
        gradients = [[0.0 for _ in range(encoder.dim)] for _ in range(encoder.dim)]
        for grad_vector, feature_vector in [
            (grad_q, features[0]),
            (grad_p, features[1]),
            (grad_n, features[2]),
        ]:
            for row_index, grad_value in enumerate(grad_vector):
                for column_index, feature_value in enumerate(feature_vector):
                    gradients[row_index][column_index] += grad_value * feature_value
        return loss, gradients
