"""Tests for the result schema contract."""

from __future__ import annotations

import json

from failurelens.schemas import (
    SCHEMA_VERSION,
    AnalysisStatus,
    Candidate,
    DiagnosisResult,
    ScoreType,
)


def test_diagnosis_result_round_trips_through_json() -> None:
    result = DiagnosisResult(
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
                score_type=ScoreType.HEURISTIC,
            )
        ],
        analysis_method="baseline-stacktrace-lexical",
    )
    payload = json.loads(result.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"] == "ok"
    assert payload["candidates"][0]["function"] == "divide"
    assert payload["candidates"][0]["score_type"] == "heuristic"

    restored = DiagnosisResult.model_validate(payload)
    assert restored == result


def test_missing_location_is_null_not_fabricated() -> None:
    candidate = Candidate(rank=1, score=0.1)
    payload = json.loads(candidate.model_dump_json())
    assert payload["file"] is None
    assert payload["start_line"] is None
    assert payload["end_line"] is None


def test_status_values_are_stable_strings() -> None:
    assert AnalysisStatus.OK == "ok"
    assert AnalysisStatus.NO_CANDIDATES == "no_candidates"
    assert AnalysisStatus.INSUFFICIENT_EVIDENCE == "insufficient_evidence"
    assert AnalysisStatus.ERROR == "error"
