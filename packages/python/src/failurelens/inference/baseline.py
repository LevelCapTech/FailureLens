"""The baseline analyzer: deterministic, dependency-free fault localization.

Pipeline (see docs/architecture.md):

* **Phase A** log parsing is performed by the caller (:mod:`failurelens.api`).
* **Phase B** source analysis is the repository index passed in.
* **Phase C/D** retrieval + ranking happen here.
* **Phase E/F** evidence collection and structured result construction happen here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from failurelens.config import DiagnoseConfig
from failurelens.indexing.base import RepositoryIndex
from failurelens.ranking.baseline import rank_candidates
from failurelens.retrieval.baseline import resolve_frames, retrieve_candidates
from failurelens.schemas import AnalysisStatus, Candidate, DiagnosisResult, ParsedLog


def _make_source_reader(root: Path) -> Callable[[str], str | None]:
    cache: dict[str, str | None] = {}

    def reader(relative_path: str) -> str | None:
        if relative_path in cache:
            return cache[relative_path]
        try:
            text: str | None = (root / relative_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = None
        cache[relative_path] = text
        return text

    return reader


class BaselineAnalyzer:
    """Stack-trace + lexical fault localization with no model or network."""

    analysis_method = "baseline-stacktrace-lexical"
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
        raw_candidates = retrieve_candidates(frames, index)

        if not raw_candidates:
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

        source_reader = (
            _make_source_reader(Path(index.root)) if config.read_candidate_source else None
        )
        candidates = rank_candidates(
            raw_candidates,
            frame_count=len(frames),
            error_message=parsed_log.error_message,
            source_reader=source_reader,
            top_k=config.top_k,
        )

        return self._result(
            repository=repository,
            parsed_log=parsed_log,
            status=AnalysisStatus.OK,
            candidates=candidates,
            warnings=warnings,
        )
