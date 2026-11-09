"""Stack-trace + static call-graph analyzer: deterministic, dependency-free, no model.

Every function and method of the repository is a candidate. A candidate scores
from two signals only (weights fixed from the v0.4 development experiment, not
re-tuned here; see ``docs/releases/v0.4/selection.md`` and ``docs/releases/v0.5``):

* **stack** - how deep in the call stack the function appears (the same formula as
  the stack-trace baseline), including the deepest in-repository frame;
* **call relation** - the function calls, or is called by (matched by name), a
  function that appears in the stack, within two hops; nearer means higher.

``score = 0.25 * stack + 0.25 * call_relation``. The score is a *heuristic*, not a
probability. Lexical overlap with the error message is reported as evidence for
the returned candidates but does not change the ranking. No learned model is used.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from failurelens.config import DiagnoseConfig
from failurelens.indexing.base import CodeUnit, RepositoryIndex
from failurelens.ranking.baseline import extract_error_tokens
from failurelens.retrieval.baseline import resolve_frames
from failurelens.retrieval.callgraph import (
    HOP_DECAY,
    CallGraph,
    Reach,
    build_call_graph,
    reach_from,
)
from failurelens.schemas import (
    AnalysisStatus,
    Candidate,
    DiagnosisResult,
    Evidence,
    EvidenceKind,
    ParsedLog,
    ScoreType,
    StackFrame,
    UnitKind,
)

WEIGHT_STACK = 0.25
WEIGHT_CALL_RELATION = 0.25
_DEPTH_WEIGHT = 0.6
_SITE_WEIGHT = 0.2
_LEXICAL_EVIDENCE_CAP = 3


def unit_id(unit: CodeUnit) -> str:
    """Stable identifier, also the deterministic tie-break (ascending)."""
    return f"{unit.file}::{unit.name}:{unit.start_line}-{unit.end_line}"


@dataclass
class _Sources:
    """Reads each file once and slices unit spans."""

    root: Path

    def __post_init__(self) -> None:
        self._lines: dict[str, list[str] | None] = {}

    def lines(self, relative_path: str) -> list[str] | None:
        if relative_path not in self._lines:
            try:
                text = (self.root / relative_path).read_text(encoding="utf-8")
                self._lines[relative_path] = text.splitlines(keepends=True)
            except (OSError, UnicodeDecodeError):
                self._lines[relative_path] = None
        return self._lines[relative_path]

    def span(self, unit: CodeUnit) -> str | None:
        lines = self.lines(unit.file)
        if lines is None or unit.end_line > len(lines):
            return None
        return "".join(lines[unit.start_line - 1 : unit.end_line])


def _function_units(index: RepositoryIndex) -> list[CodeUnit]:
    units = [
        unit
        for file_index in index.files
        for unit in file_index.units
        if unit.kind in {UnitKind.FUNCTION, UnitKind.METHOD}
    ]
    return sorted(units, key=unit_id)


@dataclass
class _Stack:
    """Stack-trace signal mapped onto function/method units."""

    scores: dict[int, float]
    deepest_by_unit: dict[int, int]
    frames_by_unit: dict[int, list[StackFrame]]
    file_level: dict[str, tuple[int, StackFrame]]
    deepest: int | None


def _stack_scores(frames: list[StackFrame], units: list[CodeUnit]) -> _Stack:
    by_file: dict[str, list[int]] = defaultdict(list)
    for position, unit in enumerate(units):
        by_file[unit.file].append(position)
    resolved = [f for f in frames if f.in_repository and f.repository_path]
    if not resolved:
        return _Stack({}, {}, {}, {}, None)
    deepest = max(f.order for f in resolved)
    denominator = max(len(frames), 1)
    deepest_by_unit: dict[int, int] = {}
    frames_by_unit: dict[int, list[StackFrame]] = defaultdict(list)
    file_level: dict[str, tuple[int, StackFrame]] = {}
    for frame in resolved:
        assert frame.repository_path is not None
        best = -1
        if frame.line is not None:
            for position in by_file.get(frame.repository_path, []):
                unit = units[position]
                if unit.contains(frame.line) and (best < 0 or unit.span() < units[best].span()):
                    best = position
        if best >= 0:
            deepest_by_unit[best] = max(frame.order, deepest_by_unit.get(best, -1))
            frames_by_unit[best].append(frame)
        else:
            known = file_level.get(frame.repository_path)
            if known is None or frame.order > known[0]:
                file_level[frame.repository_path] = (frame.order, frame)
    scores = {
        position: (
            _DEPTH_WEIGHT * ((order + 1) / denominator)
            + _SITE_WEIGHT * (1.0 if order == deepest else 0.0)
        )
        / (_DEPTH_WEIGHT + _SITE_WEIGHT)
        for position, order in deepest_by_unit.items()
    }
    return _Stack(scores, deepest_by_unit, dict(frames_by_unit), file_level, deepest)


class CallGraphAnalyzer:
    """Fault localization from the stack and static call relations; no model, no network."""

    analysis_method = "baseline-stacktrace-callgraph"
    model_id: str | None = None
    model_revision: str | None = None

    def _result(
        self,
        *,
        repository: str,
        parsed_log: ParsedLog,
        status: AnalysisStatus,
        candidates: list[Candidate],
        warnings: list[str],
    ) -> DiagnosisResult:
        return DiagnosisResult(
            status=status,
            repository=repository,
            error_type=parsed_log.error_type,
            error_message=parsed_log.error_message,
            candidates=candidates,
            analysis_method=self.analysis_method,
            model_id=self.model_id,
            model_revision=self.model_revision,
            warnings=warnings,
        )

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
        reach = reach_from(graph, sorted(stack.scores))

        scored: list[tuple[float, str, int | None, str | None]] = []
        for position, signal in stack.scores.items():
            scored.append((WEIGHT_STACK * signal, unit_id(units[position]), position, None))
        for position, hit in reach.items():
            if hit.hops == 0:
                continue
            scored.append(
                (
                    WEIGHT_CALL_RELATION * HOP_DECAY[hit.hops],
                    unit_id(units[position]),
                    position,
                    None,
                )
            )
        for path, (order, _frame) in stack.file_level.items():
            site = 1.0 if order == deepest else 0.0
            signal = (_DEPTH_WEIGHT * ((order + 1) / max(len(frames), 1)) + _SITE_WEIGHT * site) / (
                _DEPTH_WEIGHT + _SITE_WEIGHT
            )
            scored.append((WEIGHT_STACK * signal, f"{path}::<file>:0-0", None, path))
        scored.sort(key=lambda row: (-row[0], row[1]))

        tokens = extract_error_tokens(parsed_log.error_message)
        candidates: list[Candidate] = []
        for rank, (score, _identifier, row_position, row_path) in enumerate(
            scored[: config.top_k], start=1
        ):
            if row_position is None:
                assert row_path is not None
                path = row_path
                order, frame = stack.file_level[path]
                candidates.append(
                    Candidate(
                        rank=rank,
                        file=path,
                        function=None,
                        start_line=None,
                        end_line=None,
                        score=round(score, 6),
                        score_type=ScoreType.HEURISTIC,
                        evidence=self._stack_evidence(frame, order, deepest, path),
                    )
                )
                continue
            position = row_position
            unit = units[position]
            evidence = self._unit_evidence(unit, position, stack, reach.get(position), graph)
            if config.read_candidate_source:
                evidence.extend(self._lexical_evidence(unit, tokens, sources))
            candidates.append(
                Candidate(
                    rank=rank,
                    file=unit.file,
                    function=unit.name,
                    start_line=unit.start_line,
                    end_line=unit.end_line,
                    score=round(score, 6),
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

    # ------------------------------------------------------------------ evidence
    @staticmethod
    def _stack_evidence(
        frame: StackFrame, order: int, deepest: int | None, path: str
    ) -> list[Evidence]:
        location = f"{path}:{frame.line}"
        if frame.function:
            location += f" in {frame.function}"
        evidence = [
            Evidence(
                kind=EvidenceKind.STACK_FRAME,
                description=f"Appears in stack frame {order} ({location}).",
                frame_order=order,
            )
        ]
        if order == deepest:
            evidence.append(
                Evidence(
                    kind=EvidenceKind.EXCEPTION_SITE,
                    description=(
                        "Deepest in-repository frame (closest to where the error surfaced)."
                    ),
                    frame_order=order,
                )
            )
        return evidence

    def _unit_evidence(
        self,
        unit: CodeUnit,
        position: int,
        stack: _Stack,
        hit: Reach | None,
        graph: CallGraph,
    ) -> list[Evidence]:
        deepest_by_unit = stack.deepest_by_unit
        if position in deepest_by_unit:
            evidence: list[Evidence] = []
            for frame in sorted(stack.frames_by_unit[position], key=lambda f: f.order):
                location = f"{unit.file}:{frame.line}"
                if frame.function:
                    location += f" in {frame.function}"
                evidence.append(
                    Evidence(
                        kind=EvidenceKind.STACK_FRAME,
                        description=f"Appears in stack frame {frame.order} ({location}).",
                        frame_order=frame.order,
                    )
                )
            if deepest_by_unit[position] == stack.deepest:
                evidence.append(
                    Evidence(
                        kind=EvidenceKind.EXCEPTION_SITE,
                        description=(
                            "Deepest in-repository frame (closest to where the error surfaced)."
                        ),
                        frame_order=deepest_by_unit[position],
                    )
                )
            return evidence
        assert hit is not None
        origin = graph.units[hit.origin]
        origin_order = deepest_by_unit[hit.origin]
        if hit.hops == 1:
            calls = hit.origin in graph.calls[position]
            called_by = position in graph.calls[hit.origin]
            if calls and called_by:
                relation = "calls and is called by"
            elif calls:
                relation = "calls"
            else:
                relation = "is called by"
            text = (
                f"One call-relation hop (matched by name): {relation} "
                f"'{origin.qualified_name}', which appears in stack frame {origin_order}."
            )
        else:
            text = (
                f"Two call-relation hops (matched by name) from '{origin.qualified_name}', "
                f"which appears in stack frame {origin_order}."
            )
        return [
            Evidence(kind=EvidenceKind.CALL_RELATION, description=text, frame_order=origin_order)
        ]

    @staticmethod
    def _lexical_evidence(unit: CodeUnit, tokens: set[str], sources: _Sources) -> list[Evidence]:
        if not tokens:
            return []
        span = sources.span(unit)
        if span is None:
            return []
        matched = sorted(token for token in tokens if token in span)
        return [
            Evidence(
                kind=EvidenceKind.LEXICAL_MATCH,
                description=f"Error token {token!r} occurs in the candidate's source.",
            )
            for token in matched[:_LEXICAL_EVIDENCE_CAP]
        ]
