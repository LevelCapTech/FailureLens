"""Structured data models for FailureLens analysis results.

These Pydantic models define the *contract* returned by both the Python API and
the CLI (as JSON). The schema is versioned via :data:`SCHEMA_VERSION` so that
consumers can manage compatibility over time.

Design notes
------------
* ``score`` is **not** a probability. Until proper probability calibration is
  implemented and validated, scores are relative, heuristic quantities and the
  :class:`ScoreType` makes the interpretation explicit.
* When a location cannot be determined, the corresponding field is ``None``
  rather than a fabricated value (e.g. a guessed line number).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

#: Version of the result schema. Bump on any breaking change to the models below.
SCHEMA_VERSION = "0.1.0"


class AnalysisStatus(StrEnum):
    """Outcome of a diagnosis, distinguishing *inconclusive* from *failed*."""

    OK = "ok"
    """Analysis completed and produced at least one candidate."""

    NO_CANDIDATES = "no_candidates"
    """Analysis ran, but no in-repository candidate could be identified."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    """The evidence (e.g. the log) could not be parsed into anything actionable."""

    ERROR = "error"
    """Analysis could not be completed due to an unexpected error."""


class ScoreType(StrEnum):
    """How a candidate ``score`` should be interpreted."""

    HEURISTIC = "heuristic"
    """A relative, rule-based score. Not a probability. Higher means more likely."""

    SIMILARITY = "similarity"
    """A model-derived similarity (e.g. cosine). Not a calibrated probability."""


class EvidenceKind(StrEnum):
    """Category of a piece of supporting evidence for a candidate."""

    STACK_FRAME = "stack_frame"
    EXCEPTION_SITE = "exception_site"
    PATH_MATCH = "path_match"
    LEXICAL_MATCH = "lexical_match"


class UnitKind(StrEnum):
    """Kind of code unit identified during indexing."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class StackFrame(BaseModel):
    """A single frame extracted from a stack trace.

    ``file``/``line``/``function`` are reported exactly as they appear in the
    log. ``in_repository`` and ``repository_path`` are populated by the analyzer
    when a frame can be resolved to a file inside the target repository.
    """

    model_config = ConfigDict(extra="forbid")

    order: int = Field(description="0-based position in the trace; 0 is the outermost frame.")
    file: str | None = Field(default=None, description="File path as written in the trace.")
    line: int | None = Field(default=None, description="1-based line number, if present.")
    function: str | None = Field(default=None, description="Function or method name.")
    in_repository: bool = Field(
        default=False, description="True if the frame was resolved inside the target repo."
    )
    repository_path: str | None = Field(
        default=None, description="Repo-relative POSIX path when resolved; otherwise None."
    )


class ParsedLog(BaseModel):
    """Structured representation of a failure log."""

    model_config = ConfigDict(extra="forbid")

    parser: str = Field(description="Name of the parser that produced this result.")
    language: str | None = Field(default=None, description="Detected language, if known.")
    error_type: str | None = Field(default=None, description="Exception/error class name.")
    error_message: str | None = Field(default=None, description="Human-readable error message.")
    frames: list[StackFrame] = Field(
        default_factory=list, description="Stack frames ordered outermost (0) to innermost."
    )


class Evidence(BaseModel):
    """A concrete, observed reason supporting a candidate.

    Evidence describes *facts* found in the log or source (not speculative
    natural-language explanations), so results remain auditable.
    """

    model_config = ConfigDict(extra="forbid")

    kind: EvidenceKind
    description: str = Field(description="Short, factual description of the evidence.")
    frame_order: int | None = Field(
        default=None, description="Related stack-frame order, when applicable."
    )


class Candidate(BaseModel):
    """A ranked location that may be responsible for the failure."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(description="1-based rank; 1 is the most likely candidate.")
    file: str | None = Field(default=None, description="Repo-relative POSIX path, or None.")
    function: str | None = Field(default=None, description="Enclosing function/method, or None.")
    start_line: int | None = Field(default=None, description="1-based start line, or None.")
    end_line: int | None = Field(default=None, description="1-based end line, or None.")
    score: float = Field(description="Relative ranking score (see score_type).")
    score_type: ScoreType = Field(default=ScoreType.HEURISTIC)
    evidence: list[Evidence] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    """Top-level result returned by :meth:`failurelens.FailureLens.diagnose`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    status: AnalysisStatus
    repository: str | None = Field(default=None, description="Path to the analyzed repository.")
    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    candidates: list[Candidate] = Field(default_factory=list)
    analysis_method: str = Field(description="Identifier of the analysis backend used.")
    model_id: str | None = Field(
        default=None, description="Model identifier, or None for pure baseline analysis."
    )
    model_revision: str | None = Field(
        default=None, description="Pinned model revision, when a model is used."
    )
    execution_time_ms: float = Field(default=0.0, description="Wall-clock analysis time in ms.")
    warnings: list[str] = Field(default_factory=list)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a stable JSON string."""
        return self.model_dump_json(indent=indent)
