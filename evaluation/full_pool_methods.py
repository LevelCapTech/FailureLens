"""Scoring components for the v0.4 full-repository experiment (evaluator-only).

Everything here is deterministic and works on one ``RankRequest`` whose candidates
are all functions/methods of a repository. Nothing reads labels or private data.
The definitions follow ``docs/releases/v0.4/selection.md`` exactly.
"""

from __future__ import annotations

import ast
import keyword
import math
import re
import textwrap
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evaluation.private_adapter import Candidate, RankRequest, RankResult

from failurelens.ranking.baseline import _STOPWORDS, extract_error_tokens
from failurelens.schemas import ParsedLog

QUERY_PREFIX = "Represent this query for searching relevant code: "
BM25_K1 = 1.2
BM25_B = 0.75
GRAPH_MAX_SAME_NAME = 8
GRAPH_DECAY = {1: 0.5, 2: 0.25}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_STOP = frozenset(keyword.kwlist) | frozenset(_STOPWORDS)

Parser = Callable[[str], ParsedLog | None]


def tokenize(text: str) -> list[str]:
    """Identifiers split on underscores and camelCase, lowercased, >=3 chars, no stopwords."""
    tokens: list[str] = []
    for identifier in _IDENT.findall(text):
        for part in identifier.split("_"):
            for piece in _CAMEL.findall(part):
                token = piece.lower()
                if len(token) >= 3 and token not in _STOP:
                    tokens.append(token)
    return tokens


def error_log(request: RankRequest) -> str:
    import json

    framed = json.loads(request.query)
    return str(framed["error_log"])


def bm25_scores(log: str, candidates: Sequence[Candidate]) -> list[float]:
    """BM25 (k1=1.2, b=0.75) of the log's distinct tokens against each candidate's source."""
    query = set(tokenize(log))
    documents = [tokenize(c.source_text) for c in candidates]
    size = len(documents)
    if not query or size == 0:
        return [0.0] * size
    lengths = [len(doc) for doc in documents]
    average = (sum(lengths) / size) or 1.0
    frequency: dict[str, int] = defaultdict(int)
    counts: list[dict[str, int]] = []
    for doc in documents:
        local: dict[str, int] = defaultdict(int)
        for token in doc:
            local[token] += 1
        counts.append(local)
        for token in local:
            if token in query:
                frequency[token] += 1
    scores: list[float] = []
    for local, length in zip(counts, lengths, strict=True):
        score = 0.0
        for token in query:
            tf = local.get(token, 0)
            if not tf:
                continue
            df = frequency[token]
            idf = math.log(1 + (size - df + 0.5) / (df + 0.5))
            score += (
                idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / average))
            )
        scores.append(score)
    return scores


def capped_lexical_scores(log: str, candidates: Sequence[Candidate]) -> list[float]:
    """The shipped baseline's lexical feature: error tokens contained in the source, capped at 3."""
    tokens = extract_error_tokens(log)
    return [min(sum(token in c.source_text for token in tokens), 3) / 3 for c in candidates]


def _normalize_max(values: Sequence[float]) -> list[float]:
    top = max(values, default=0.0)
    return [v / top for v in values] if top > 0 else [0.0] * len(values)


def _normalize_minmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 0:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


@dataclass(frozen=True)
class Located:
    file: str
    name: str
    start: int
    end: int


def locate(candidate: Candidate) -> Located:
    """Parse ``file::name:start-end`` (the evaluator's candidate ID)."""
    head, span = candidate.id.rsplit(":", 1)
    file, name = head.split("::", 1)
    start, end = span.split("-", 1)
    return Located(file, name, int(start), int(end))


def _resolve_path(frame_file: str | None, paths: set[str]) -> str | None:
    if not frame_file:
        return None
    parts = [p for p in frame_file.replace("\\", "/").split("/") if p not in ("", ".")]
    for start in range(len(parts)):
        suffix = "/".join(parts[start:])
        if suffix in paths:
            return suffix
    if parts:
        matches = [p for p in paths if p.rsplit("/", 1)[-1] == parts[-1]]
        if len(matches) == 1:
            return matches[0]
    return None


def stack_signal(
    log: str, candidates: Sequence[Candidate], parse: Parser
) -> tuple[list[float], set[int]]:
    """Per-candidate stack score in [0, 1] and the indices of stack-frame functions."""
    parsed = parse(log)
    frames = parsed.frames if parsed is not None else []
    located = [locate(c) for c in candidates]
    paths = {item.file for item in located}
    by_file: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(located):
        by_file[item.file].append(index)
    resolved: list[tuple[int, int]] = []  # (frame order, candidate index)
    for frame in frames:
        path = _resolve_path(frame.file, paths)
        if path is None:
            continue
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
    scores = [0.0] * len(candidates)
    seeds: set[int] = set()
    if not resolved:
        return scores, seeds
    deepest = max(order for order, _ in resolved)
    denominator = max(len(frames), 1)
    deepest_by_candidate: dict[int, int] = {}
    for order, index in resolved:
        if index >= 0:
            deepest_by_candidate[index] = max(order, deepest_by_candidate.get(index, -1))
    for index, order in deepest_by_candidate.items():
        site = 1.0 if order == deepest else 0.0
        scores[index] = (0.6 * ((order + 1) / denominator) + 0.2 * site) / 0.8
        seeds.add(index)
    return scores, seeds


def _called_names(source: str) -> set[str]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def call_graph(candidates: Sequence[Candidate]) -> list[set[int]]:
    """Undirected static call graph over candidates, linking callers and same-name callees."""
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_name[locate(candidate).name].append(index)
    adjacency: list[set[int]] = [set() for _ in candidates]
    for index, candidate in enumerate(candidates):
        for name in _called_names(candidate.source_text):
            targets = by_name.get(name, [])
            if not targets or len(targets) > GRAPH_MAX_SAME_NAME:
                continue
            for target in targets:
                if target != index:
                    adjacency[index].add(target)
                    adjacency[target].add(index)
    return adjacency


def graph_signal(adjacency: Sequence[set[int]], seeds: set[int]) -> list[float]:
    """0.5 for functions one call away from a stack function, 0.25 for two; seeds get 0."""
    scores = [0.0] * len(adjacency)
    if not seeds:
        return scores
    distance = dict.fromkeys(seeds, 0)
    queue = deque(seeds)
    while queue:
        node = queue.popleft()
        if distance[node] >= 2:
            continue
        for neighbour in sorted(adjacency[node]):
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    for node, hops in distance.items():
        scores[node] = GRAPH_DECAY.get(hops, 0.0)
    return scores


@dataclass(frozen=True)
class CaseComponents:
    """Per-case normalized signals; the embedding part is attached per model."""

    ids: tuple[str, ...]
    lexical: tuple[float, ...]  # BM25 / max
    capped: tuple[float, ...]
    stack: tuple[float, ...]
    graph: tuple[float, ...]


def build_components(request: RankRequest, parse: Parser) -> CaseComponents:
    log = error_log(request)
    candidates = request.candidates
    stack, seeds = stack_signal(log, candidates, parse)
    graph = graph_signal(call_graph(candidates), seeds)
    return CaseComponents(
        tuple(c.id for c in candidates),
        tuple(_normalize_max(bm25_scores(log, candidates))),
        tuple(capped_lexical_scores(log, candidates)),
        tuple(stack),
        tuple(graph),
    )


def order_by(ids: Sequence[str], scores: Sequence[float]) -> tuple[str, ...]:
    """Descending score; ties by candidate ID ascending."""
    ranked = sorted(zip(ids, scores, strict=True), key=lambda row: (-row[1], row[0]))
    return tuple(identifier for identifier, _ in ranked)


def fused_scores(
    components: CaseComponents,
    similarity: Sequence[float],
    weights: tuple[float, float, float, float],
) -> list[float]:
    """``w_e*E + w_l*L + w_s*s + w_g*g`` with E min-max normalized over the pool."""
    w_e, w_l, w_s, w_g = weights
    embedded = _normalize_minmax(similarity)
    return [
        w_e * e + w_l * lex + w_s * s + w_g * g
        for e, lex, s, g in zip(
            embedded, components.lexical, components.stack, components.graph, strict=True
        )
    ]


def result_for(request: RankRequest, ordered: Sequence[str]) -> RankResult:
    return RankResult(tuple(ordered), request.digest, request.top_k)
