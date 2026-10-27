"""Tests for the high-level FailureLens API."""

from __future__ import annotations

from pathlib import Path

import pytest

from failurelens import FailureLens
from failurelens.errors import InputError
from failurelens.schemas import AnalysisStatus, ScoreType


def test_diagnose_from_error_text_ranks_exception_site_first(
    sample_repo: Path, sample_traceback: str
) -> None:
    lens = FailureLens.from_baseline()
    result = lens.diagnose(sample_repo, error_text=sample_traceback, top_k=5)

    assert result.status is AnalysisStatus.OK
    assert result.error_type == "ZeroDivisionError"
    assert result.analysis_method == "baseline-stacktrace-lexical"
    assert result.model_id is None
    assert result.execution_time_ms >= 0.0

    top = result.candidates[0]
    assert top.file == "pkg/core.py"
    assert top.function == "divide"
    assert top.score_type is ScoreType.HEURISTIC
    # The root-cause function is also surfaced among the candidates.
    assert any(c.function == "average" for c in result.candidates)


def test_diagnose_from_log_file(sample_repo: Path, sample_traceback: str, tmp_path: Path) -> None:
    log = tmp_path / "failure.log"
    log.write_text(sample_traceback, encoding="utf-8")
    lens = FailureLens.from_baseline()
    result = lens.diagnose(sample_repo, error_log=log)
    assert result.status is AnalysisStatus.OK


def test_diagnose_missing_log_file_raises_input_error(sample_repo: Path) -> None:
    lens = FailureLens.from_baseline()
    with pytest.raises(InputError):
        lens.diagnose(sample_repo, error_log=sample_repo / "nope.log")


def test_diagnose_requires_exactly_one_log_source(sample_repo: Path, sample_traceback: str) -> None:
    lens = FailureLens.from_baseline()
    with pytest.raises(InputError):
        lens.diagnose(sample_repo)  # neither error_log nor error_text
    with pytest.raises(InputError):
        lens.diagnose(sample_repo, error_log=sample_repo / "x", error_text=sample_traceback)


def test_diagnose_reuses_prebuilt_index(sample_repo: Path, sample_traceback: str) -> None:
    lens = FailureLens.from_baseline()
    index = lens.index(sample_repo)
    result = lens.diagnose(sample_repo, error_text=sample_traceback, index=index)
    assert result.status is AnalysisStatus.OK


def test_diagnose_unparseable_log_is_insufficient_evidence(sample_repo: Path) -> None:
    lens = FailureLens.from_baseline()
    result = lens.diagnose(sample_repo, error_text="not a traceback, just noise")
    assert result.status is AnalysisStatus.INSUFFICIENT_EVIDENCE
    assert result.warnings


def test_diagnose_no_repo_frames_is_no_candidates(sample_repo: Path) -> None:
    lens = FailureLens.from_baseline()
    text = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.11/json/decoder.py", line 1, in decode\n'
        "    raise ValueError\n"
        "ValueError: bad\n"
    )
    result = lens.diagnose(sample_repo, error_text=text)
    assert result.status is AnalysisStatus.NO_CANDIDATES


def test_diagnose_respects_top_k(sample_repo: Path, sample_traceback: str) -> None:
    lens = FailureLens.from_baseline()
    result = lens.diagnose(sample_repo, error_text=sample_traceback, top_k=1)
    assert len(result.candidates) == 1
