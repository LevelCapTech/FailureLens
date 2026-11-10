"""Stack + call-graph analyzer with fixed, learned linear weights (opt-in, no model at run time).

Every function and method of the repository is scored by a logistic-regression
decision value over 15 deterministic features (stack position, call relations,
test-file location, name/log/message matches). The constants come from
``weighted_constants`` (frozen v0.7 candidate D, fitted offline on development
data; see ``docs/releases/v0.7``). Nothing is learned or downloaded at run time,
the analyzer is dependency-free and offline, and the score is a *heuristic*,
not a probability. It is **not** the default strategy.
"""

from __future__ import annotations

import keyword
import re
from pathlib import Path

from failurelens.config import DiagnoseConfig
from failurelens.indexing.base import CodeUnit, RepositoryIndex
from failurelens.inference import weighted_constants as constants
from failurelens.inference.callgraph import CallGraphAnalyzer, _function_units, _Sources, unit_id
from failurelens.ranking.baseline import _STOPWORDS, extract_error_tokens
from failurelens.retrieval.baseline import resolve_frames
from failurelens.retrieval.callgraph import CallGraph, build_call_graph, reach_from
from failurelens.schemas import (
    AnalysisStatus,
    Candidate,
    DiagnosisResult,
    Evidence,
    EvidenceKind,
    ParsedLog,
    ScoreType,
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_STOP = frozenset(keyword.kwlist) | frozenset(_STOPWORDS)
_METHOD = re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\(\s*(self|cls)\b")
_TEST_DIRS = frozenset({"tests", "test", "testing"})
_NEAR_LINES = 100
_UNREACHED_HOPS = 3


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for identifier in _IDENT.findall(text):
        for part in identifier.split("_"):
            for piece in _CAMEL.findall(part):
                token = piece.lower()
                if len(token) >= 3 and token not in _STOP:
                    tokens.append(token)
    return tokens


def is_test_file(path: str) -> bool:
    """Whether a repository-relative path looks like test code (decided from the path only)."""
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return (
        any(part in _TEST_DIRS for part in parts[:-1])
        or name == "conftest.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


class WeightedCallGraphAnalyzer(CallGraphAnalyzer):
    """Fault localization from stack, call relations and test-file location with fixed weights."""

    analysis_method = "baseline-stacktrace-callgraph-weighted"
    model_id: str | None = None
    model_revision: str | None = None

    def analyze(
        self,
        *,
        repository: str,
        parsed_log: ParsedLog,
        index: RepositoryIndex,
        config: DiagnoseConfig,
    ) -> DiagnosisResult:
        warnings: list[str] = list(index.warnings)
        if not parsed_log.frames:
            warnings.append("The log parsed but contained no stack frames to localize.")
            return self._result(
                repository=repository,
                parsed_log=parsed_log,
                status=AnalysisStatus.INSUFFICIENT_EVIDENCE,
                candidates=[],
                warnings=warnings,
            )
        from failurelens.inference.callgraph import _stack_scores

        frames = resolve_frames(parsed_log.frames, index)
        units = _function_units(index)
        stack = _stack_scores(frames, units)
        deepest = stack.deepest
        if deepest is None:
            warnings.append(
                "Stack frames were found but none resolved to files in the target repository."
            )
            return self._result(
                repository=repository,
                parsed_log=parsed_log,
                status=AnalysisStatus.NO_CANDIDATES,
                candidates=[],
                warnings=warnings,
            )

        sources = _Sources(Path(index.root))
        graph: CallGraph = build_call_graph(units, sources.span)
        if graph.skipped_reason:
            warnings.append(graph.skipped_reason)
        seeds = sorted(stack.scores)
        reach = reach_from(graph, seeds)
        test_flags = [is_test_file(unit.file) for unit in units]
        production_seeds = [s for s in seeds if not test_flags[s]]
        production_reach = reach_from(graph, production_seeds)

        log_text = parsed_log.raw_text or ""
        log_identifiers = set(_IDENT.findall(log_text))
        message_tokens = set(_tokenize(parsed_log.error_message or ""))
        seed_files = {units[s].file for s in seeds}
        method_starts: dict[str, list[int]] = {}
        for seed in production_seeds:
            if self._is_method(units[seed], sources):
                method_starts.setdefault(units[seed].file, []).append(units[seed].start_line)
        width = max(max(len(frames), 1) - 1, 1)
        seed_set = set(seeds)
        production_set = set(production_seeds)

        scored: list[tuple[float, str, int]] = []
        for position, unit in enumerate(units):
            is_seed = position in seed_set
            hops = reach[position].hops if position in reach else _UNREACHED_HOPS
            prod_hops = (
                production_reach[position].hops if position in production_reach else _UNREACHED_HOPS
            )
            name_tokens = _tokenize(unit.name)
            near = (
                position not in production_set
                and any(
                    abs(unit.start_line - start) <= _NEAR_LINES
                    for start in method_starts.get(unit.file, [])
                )
                and self._is_method(unit, sources)
            )
            row = (
                stack.scores.get(position, 0.0),
                float(is_seed),
                float(is_seed and stack.deepest_by_unit[position] == deepest),
                (deepest - stack.deepest_by_unit[position]) / width if is_seed else 1.0,
                float(any(position in graph.calls[s] for s in seeds if s != position)),
                float(any(s in graph.calls[position] for s in seeds if s != position)),
                float(hops == 2),
                float(hops),
                float(unit.name in log_identifiers),
                float(unit.file in seed_files),
                float(test_flags[position]),
                float(position in production_set),
                prod_hops / 3.0,
                sum(token in message_tokens for token in name_tokens) / len(name_tokens)
                if name_tokens
                else 0.0,
                float(near),
            )
            value = constants.INTERCEPT
            for x, mean, scale, weight in zip(
                row, constants.MEAN, constants.SCALE, constants.COEFFICIENTS, strict=True
            ):
                value += weight * ((x - mean) / scale)
            scored.append((value, unit_id(unit), position))
        scored.sort(key=lambda item: (-item[0], item[1]))

        tokens = extract_error_tokens(parsed_log.error_message)
        candidates: list[Candidate] = []
        for rank, (value, _identifier, position) in enumerate(scored[: config.top_k], start=1):
            unit = units[position]
            evidence = self._weighted_evidence(
                unit, position, stack, reach.get(position), graph, test_flags[position]
            )
            if config.read_candidate_source:
                evidence.extend(self._lexical_evidence(unit, tokens, sources))
            candidates.append(
                Candidate(
                    rank=rank,
                    file=unit.file,
                    function=unit.name,
                    start_line=unit.start_line,
                    end_line=unit.end_line,
                    score=round(value, 6),
                    score_type=ScoreType.HEURISTIC,
                    evidence=evidence,
                )
            )
        return self._result(
            repository=repository,
            parsed_log=parsed_log,
            status=AnalysisStatus.OK,
            candidates=candidates,
            warnings=warnings,
        )

    @staticmethod
    def _is_method(unit: CodeUnit, sources: _Sources) -> bool:
        span = sources.span(unit)
        return span is not None and _METHOD.match(span) is not None

    def _weighted_evidence(
        self,
        unit: CodeUnit,
        position: int,
        stack: object,
        hit: object,
        graph: CallGraph,
        in_test_file: bool,
    ) -> list[Evidence]:
        from failurelens.inference.callgraph import _Stack
        from failurelens.retrieval.callgraph import Reach

        assert isinstance(stack, _Stack)
        evidence: list[Evidence] = []
        if position in stack.deepest_by_unit or isinstance(hit, Reach):
            evidence = self._unit_evidence(
                unit, position, stack, hit if isinstance(hit, Reach) else None, graph
            )
        if in_test_file:
            evidence.append(
                Evidence(
                    kind=EvidenceKind.PATH_MATCH,
                    description="Located in a test file; such functions are ranked lower.",
                )
            )
        return evidence


__all__ = ["WeightedCallGraphAnalyzer", "is_test_file"]
