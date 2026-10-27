"""FailureLens: AI-powered fault localization toolkit.

FailureLens analyzes failure evidence (error logs, stack traces) against a source
repository and returns ranked candidate locations for the fault, with structured,
auditable evidence. The baseline runs on CPU with no network access and no heavy
dependencies; model-based backends are optional.

Quickstart
----------
    >>> from failurelens import FailureLens
    >>> lens = FailureLens.from_baseline()
    >>> result = lens.diagnose(repository="./project", error_log="./failure.log")
    >>> for candidate in result.candidates:
    ...     print(candidate.rank, candidate.file, candidate.function, candidate.score)
"""

from __future__ import annotations

from failurelens._version import __version__
from failurelens.api import FailureLens
from failurelens.config import DiagnoseConfig, IndexConfig
from failurelens.errors import (
    FailureLensError,
    InputError,
    MissingDependencyError,
    ModelError,
    RepositoryError,
)
from failurelens.schemas import (
    SCHEMA_VERSION,
    AnalysisStatus,
    Candidate,
    DiagnosisResult,
    Evidence,
    EvidenceKind,
    ParsedLog,
    ScoreType,
    StackFrame,
)

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisStatus",
    "Candidate",
    "DiagnoseConfig",
    "DiagnosisResult",
    "Evidence",
    "EvidenceKind",
    "FailureLens",
    "FailureLensError",
    "IndexConfig",
    "InputError",
    "MissingDependencyError",
    "ModelError",
    "ParsedLog",
    "RepositoryError",
    "ScoreType",
    "StackFrame",
    "__version__",
]
