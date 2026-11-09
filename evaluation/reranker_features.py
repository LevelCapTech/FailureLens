"""Per-candidate features for the v0.6 learned reranker (evaluator-only, deterministic).

The feature list is fixed in ``docs/releases/v0.6/selection-rule.md`` before any result.
Everything here works on one ``RankRequest`` whose candidates are all functions of a
repository. It reads no labels and no private data, uses no randomness, and the library
(``packages/python``) does not import it.

Group G1 (``f01``..``f23``) uses the log, the stack frames, the static call graph and the
function text. Group G2 (``g01``..``g02``) is an optional embedding similarity that the
caller supplies.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from evaluation.full_pool_methods import (
    _IDENT,
    GRAPH_MAX_SAME_NAME,
    Parser,
    _called_names,
    _normalize_max,
    _normalize_minmax,
    _resolve_path,
    bm25_scores,
    capped_lexical_scores,
    error_log,
    locate,
    stack_signal,
    tokenize,
)
from evaluation.private_adapter import Candidate, RankRequest

G1_NAMES: tuple[str, ...] = (
    "f01_stack_score",
    "f02_in_stack",
    "f03_deepest_frame",
    "f04_relative_depth",
    "f05_called_by_stack",
    "f06_calls_stack",
    "f07_two_hops",
    "f08_min_hops",
    "f09_stack_functions_within_2",
    "f10_bm25_max_norm",
    "f11_bm25_rank_inverse",
    "f12_capped_lexical",
    "f13_log_bm25",
    "f14_name_token_overlap",
    "f15_name_in_log",
    "f16_same_file_as_stack",
    "f17_same_dir_as_frame",
    "f18_path_token_overlap",
    "f19_error_type_in_body",
    "f20_log_lines",
    "f21_log_tokens",
    "f22_log_pool_size",
    "f23_stack_found",
)
G2_NAMES: tuple[str, ...] = ("g01_embedding_norm", "g02_embedding_rank_inverse")
MAX_HOPS = 3


@dataclass(frozen=True)
class StackInfo:
    """Which candidate hosts which stack frame (same resolution as ``stack_signal``)."""

    deepest_order_by_candidate: dict[int, int]
    deepest_order: int | None
    frame_files: frozenset[str]
    frame_count: int


def stack_info(log: str, candidates: Sequence[Candidate], parse: Parser) -> StackInfo:
    parsed = parse(log)
    frames = parsed.frames if parsed is not None else []
    located = [locate(c) for c in candidates]
    paths = {item.file for item in located}
    by_file: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(located):
        by_file[item.file].append(index)
    resolved: list[tuple[int, int]] = []
    files: set[str] = set()
    for frame in frames:
        path = _resolve_path(frame.file, paths)
        if path is None:
            continue
        files.add(path)
        resolved.append((frame.order, -1))
        if frame.line is None:
            continue
        best = -1
        for index in by_file[path]:
            item = located[index]
            if item.start <= frame.line <= item.end and (
                best < 0 or item.end - item.start < located[best].end - located[best].start
            ):
                best = index
        resolved[-1] = (frame.order, best)
    deepest_by_candidate: dict[int, int] = {}
    for order, index in resolved:
        if index >= 0:
            deepest_by_candidate[index] = max(order, deepest_by_candidate.get(index, -1))
    deepest = max((order for order, _ in resolved), default=None)
    return StackInfo(deepest_by_candidate, deepest, frozenset(files), max(len(frames), 1))


def directed_call_graph(
    candidates: Sequence[Candidate],
) -> tuple[list[set[int]], list[set[int]]]:
    """``(callees, callers)`` by name, ignoring names shared by more than 8 candidates."""
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_name[locate(candidate).name].append(index)
    callees: list[set[int]] = [set() for _ in candidates]
    callers: list[set[int]] = [set() for _ in candidates]
    for index, candidate in enumerate(candidates):
        for name in _called_names(candidate.source_text):
            targets = by_name.get(name, [])
            if not targets or len(targets) > GRAPH_MAX_SAME_NAME:
                continue
            for target in targets:
                if target != index:
                    callees[index].add(target)
                    callers[target].add(index)
    return callees, callers


def _hop_distances(adjacency: Sequence[set[int]], seeds: set[int]) -> dict[int, int]:
    distance = dict.fromkeys(seeds, 0)
    queue = deque(sorted(seeds))
    while queue:
        node = queue.popleft()
        if distance[node] >= 2:
            continue
        for neighbour in sorted(adjacency[node]):
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance


def _seeds_within_two(adjacency: Sequence[set[int]], seeds: set[int]) -> list[int]:
    counts = [0] * len(adjacency)
    for seed in sorted(seeds):
        for node, hops in _hop_distances(adjacency, {seed}).items():
            if node != seed and hops <= 2:
                counts[node] += 1
    return counts


def _rank_inverse(values: Sequence[float], ids: Sequence[str]) -> list[float]:
    """``1 / (1 + rank)`` with rank 0 for the largest value; ties by candidate ID ascending."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], ids[i]))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = 1.0 / (1.0 + rank)
    return result


@dataclass(frozen=True)
class CaseFeatures:
    ids: tuple[str, ...]
    g1: np.ndarray  # (N, 23) float64
    seeds: frozenset[int]


def case_features(request: RankRequest, parse: Parser) -> CaseFeatures:
    log = error_log(request)
    candidates = request.candidates
    ids = tuple(c.id for c in candidates)
    size = len(candidates)
    located = [locate(c) for c in candidates]
    stack_scores, seeds = stack_signal(log, candidates, parse)
    info = stack_info(log, candidates, parse)
    if set(info.deepest_order_by_candidate) != seeds:  # pragma: no cover - guards drift
        raise AssertionError("stack resolution drifted from stack_signal")
    callees, callers = directed_call_graph(candidates)
    undirected = [callees[i] | callers[i] for i in range(size)]
    distance = _hop_distances(undirected, seeds) if seeds else {}
    near = _seeds_within_two(undirected, seeds) if seeds else [0] * size
    raw_bm25 = bm25_scores(log, candidates)
    bm25_norm = _normalize_max(raw_bm25)
    bm25_inverse = _rank_inverse(raw_bm25, ids)
    capped = capped_lexical_scores(log, candidates)
    log_tokens = set(tokenize(log))
    log_identifiers = set(_IDENT.findall(log))
    parsed = parse(log)
    error_type = parsed.error_type if parsed is not None else None
    seed_files = {located[i].file for i in seeds}
    frame_dirs = {os.path.dirname(path) for path in info.frame_files}
    deepest = info.deepest_order
    width = max(info.frame_count - 1, 1)
    matrix = np.zeros((size, len(G1_NAMES)), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        item = located[index]
        is_seed = index in seeds
        hops = distance.get(index, MAX_HOPS)
        name_tokens = tokenize(item.name)
        path_tokens = tokenize(item.file.replace("/", " ").replace(".", " "))
        row = (
            stack_scores[index],
            float(is_seed),
            float(is_seed and info.deepest_order_by_candidate[index] == deepest),
            (deepest - info.deepest_order_by_candidate[index]) / width
            if is_seed and deepest is not None
            else 1.0,
            float(any(index in callees[s] for s in seeds if s != index)),
            float(any(s in callees[index] for s in seeds if s != index)),
            float(hops == 2),
            float(min(hops, MAX_HOPS)),
            math.log1p(near[index]),
            bm25_norm[index],
            bm25_inverse[index],
            capped[index],
            math.log1p(raw_bm25[index]),
            sum(token in log_tokens for token in name_tokens) / max(len(name_tokens), 1),
            float(item.name in log_identifiers),
            float(item.file in seed_files),
            float(os.path.dirname(item.file) in frame_dirs),
            sum(token in log_tokens for token in path_tokens) / max(len(path_tokens), 1),
            float(bool(error_type) and str(error_type) in candidate.source_text),
            math.log1p(item.end - item.start + 1),
            math.log1p(len(tokenize(candidate.source_text))),
            math.log1p(size),
            float(bool(seeds)),
        )
        matrix[index] = row
    return CaseFeatures(ids, matrix, frozenset(seeds))


def embedding_features(similarity: Sequence[float], ids: Sequence[str]) -> np.ndarray:
    """``(N, 2)``: min-max normalized cosine and its rank inverse (ties by ID)."""
    return np.column_stack(
        [
            np.asarray(_normalize_minmax(similarity), dtype=np.float64),
            np.asarray(_rank_inverse(similarity, ids), dtype=np.float64),
        ]
    )
