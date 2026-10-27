"""Tests for the JSON and text reporters."""

from __future__ import annotations

import json

from failurelens.reporting import to_json, to_text
from failurelens.schemas import (
    AnalysisStatus,
    Candidate,
    DiagnosisResult,
    Evidence,
    EvidenceKind,
)


def _result() -> DiagnosisResult:
    return DiagnosisResult(
        status=AnalysisStatus.OK,
        repository="/tmp/project",
        error_type="ZeroDivisionError",
        error_message="division by zero",
        candidates=[
            Candidate(
                rank=1,
                file="pkg/core.py",
                function="divide",
                start_line=4,
                end_line=6,
                score=0.8,
                evidence=[
                    Evidence(
                        kind=EvidenceKind.EXCEPTION_SITE,
                        description="Deepest in-repository frame.",
                    )
                ],
            )
        ],
        analysis_method="baseline-stacktrace-lexical",
    )


def test_to_json_is_valid_and_versioned() -> None:
    payload = json.loads(to_json(_result()))
    assert payload["schema_version"]
    assert payload["analysis_method"] == "baseline-stacktrace-lexical"


def test_to_text_is_human_readable() -> None:
    text = to_text(_result())
    assert "ZeroDivisionError" in text
    assert "pkg/core.py" in text
    assert "divide" in text
    # Evidence is shown by default.
    assert "Deepest in-repository frame." in text


def test_to_text_handles_no_candidates() -> None:
    result = DiagnosisResult(
        status=AnalysisStatus.NO_CANDIDATES,
        analysis_method="baseline-stacktrace-lexical",
    )
    text = to_text(result)
    assert "no_candidates" in text
    assert "No candidates were identified." in text
