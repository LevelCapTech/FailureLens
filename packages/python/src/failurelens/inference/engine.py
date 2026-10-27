"""Analyzer protocol shared by every analysis backend.

An analyzer consumes a parsed log and a repository index and produces a
:class:`~failurelens.schemas.DiagnosisResult`. Keeping this a small protocol lets
the baseline and any future model-based backend be swapped behind the same
public API and tested independently.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from failurelens.config import DiagnoseConfig
from failurelens.indexing.base import RepositoryIndex
from failurelens.schemas import DiagnosisResult, ParsedLog


@runtime_checkable
class Analyzer(Protocol):
    """Interface implemented by analysis backends."""

    #: Stable identifier recorded in :attr:`DiagnosisResult.analysis_method`.
    analysis_method: str
    #: Model id, or ``None`` for pure baseline analysis.
    model_id: str | None
    #: Pinned model revision, when a model is used.
    model_revision: str | None

    def analyze(
        self,
        *,
        repository: str,
        parsed_log: ParsedLog,
        index: RepositoryIndex,
        config: DiagnoseConfig,
    ) -> DiagnosisResult:
        """Produce a diagnosis from a parsed log and repository index."""
        ...
