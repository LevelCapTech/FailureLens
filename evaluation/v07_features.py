"""Per-candidate group-T features for the v0.7 experiment (evaluator-only, deterministic).

Definitions are fixed in ``docs/releases/v0.7/selection-rule.md`` before any result. Reads no
labels and no private data, uses no randomness, and the library does not import it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from evaluation.full_pool_methods import (
    Parser,
    call_graph,
    error_log,
    graph_signal,
    locate,
    stack_signal,
    tokenize,
)
from evaluation.private_adapter import Candidate, RankRequest
from evaluation.reranker_features import CaseFeatures, _hop_distances, directed_call_graph

T_NAMES: tuple[str, ...] = (
    "t01_is_test_file",
    "t02_production_stack",
    "t03_production_hops",
    "t04_calls_production_stack",
    "t05_called_by_production_stack",
    "t06_name_in_message",
    "t07_same_class_near",
)
# columns of the 23-column G1 matrix used by the learned candidates (f01-f08, f15, f16)
G1_COLUMNS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 14, 15)
W_STACK = 0.25
W_GRAPH = 0.25
W_MESSAGE = 0.15
NEAR_LINES = 100
_TEST_DIRS = frozenset({"tests", "test", "testing"})
_METHOD = re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\(\s*(self|cls)\b")


def is_test_file(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return (
        any(p in _TEST_DIRS for p in parts[:-1])
        or name == "conftest.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _is_method(candidate: Candidate) -> bool:
    return _METHOD.match(candidate.source_text) is not None


@dataclass(frozen=True)
class CaseV07:
    ids: tuple[str, ...]
    t: np.ndarray  # (N, 7) in T_NAMES order
    s5: np.ndarray  # V5 score
    s5_prod: np.ndarray  # rule-B score
    message: np.ndarray  # t06 (kept separately for rule C)
    is_test: np.ndarray  # bool


def case_v07(request: RankRequest, parse: Parser) -> CaseV07:
    log = error_log(request)
    candidates = request.candidates
    size = len(candidates)
    located = [locate(c) for c in candidates]
    is_test = np.array([is_test_file(item.file) for item in located], dtype=bool)
    stack, seeds = stack_signal(log, candidates, parse)
    undirected = call_graph(candidates)
    base_graph = graph_signal(undirected, seeds)
    s5 = np.array([W_STACK * s + W_GRAPH * g for s, g in zip(stack, base_graph, strict=True)])
    prod_seeds = {i for i in seeds if not is_test[i]}
    stack_prod = [0.0 if is_test[i] else s for i, s in enumerate(stack)]
    graph_prod = graph_signal(undirected, prod_seeds)
    s5_prod = np.array(
        [W_STACK * s + W_GRAPH * g for s, g in zip(stack_prod, graph_prod, strict=True)]
    )

    callees, _callers = directed_call_graph(candidates)
    distance = _hop_distances(undirected, prod_seeds) if prod_seeds else {}
    parsed = parse(log)
    message_tokens = set(tokenize(parsed.error_message or "")) if parsed is not None else set()
    methods = [_is_method(c) for c in candidates]
    prod_method_starts: dict[str, list[int]] = {}
    for i in prod_seeds:
        if methods[i]:
            prod_method_starts.setdefault(located[i].file, []).append(located[i].start)
    t = np.zeros((size, len(T_NAMES)), dtype=np.float64)
    for i in range(size):
        name_tokens = tokenize(located[i].name)
        t06 = (
            sum(token in message_tokens for token in name_tokens) / len(name_tokens)
            if name_tokens
            else 0.0
        )
        near = (
            methods[i]
            and i not in prod_seeds
            and any(
                abs(located[i].start - start) <= NEAR_LINES
                for start in prod_method_starts.get(located[i].file, [])
            )
        )
        t[i] = (
            float(is_test[i]),
            float(i in prod_seeds),
            min(distance.get(i, 3), 3) / 3.0,
            float(any(s in callees[i] for s in prod_seeds if s != i)),
            float(any(i in callees[s] for s in prod_seeds if s != i)),
            t06,
            float(near),
        )
    return CaseV07(tuple(c.id for c in candidates), t, s5, s5_prod, t[:, 5].copy(), is_test)


def rule_scores(case: CaseV07, key: str) -> Sequence[float]:
    """Sort score for the rule candidates A, B, C (test files last, then descending score)."""
    penalty = np.where(case.is_test, -1000.0, 0.0)
    if key == "A":
        return list(case.s5 + penalty)
    if key == "B":
        return list(case.s5_prod + penalty)
    if key == "C":
        return list(case.s5_prod + W_MESSAGE * case.message + penalty)
    raise ValueError(key)


def learned_matrix(f: CaseFeatures, c: CaseV07) -> np.ndarray:
    """The 15 learned-candidate features: f01-f08, f15, f16, t01, t02, t03, t06, t07."""
    t_cols = (0, 1, 2, 5, 6)
    return np.hstack([f.g1[:, list(G1_COLUMNS)], c.t[:, list(t_cols)]])
