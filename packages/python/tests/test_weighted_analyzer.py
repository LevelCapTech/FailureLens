"""Tests for the opt-in weighted call-graph analyzer (frozen v0.7 candidate D)."""

from __future__ import annotations

from pathlib import Path

import pytest

from failurelens import FailureLens
from failurelens.inference import (
    BASELINE_STRATEGIES,
    DEFAULT_BASELINE_STRATEGY,
    WeightedCallGraphAnalyzer,
)
from failurelens.inference import weighted_constants as constants
from failurelens.inference.weighted import is_test_file
from failurelens.schemas import AnalysisStatus, EvidenceKind, ScoreType

SERVICE_PY = """\
def normalize(value):
    return value.strip()


def validate(value):
    return normalize(value)


def handle(request):
    return validate(request).lower()
"""

TEST_PY = """\
from service import handle


def test_handle():
    return handle(None)
"""

TRACEBACK = """\
Traceback (most recent call last):
  File "tests/test_service.py", line 5, in test_handle
    return handle(None)
  File "service.py", line 9, in handle
    return validate(request).lower()
AttributeError: 'NoneType' object has no attribute 'strip'
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "service.py").write_text(SERVICE_PY, encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_service.py").write_text(TEST_PY, encoding="utf-8")
    return tmp_path


def _lens() -> FailureLens:
    return FailureLens.from_baseline(strategy="stacktrace-callgraph-weighted")


def test_registered_not_default_and_honestly_labelled() -> None:
    assert BASELINE_STRATEGIES["stacktrace-callgraph-weighted"] is WeightedCallGraphAnalyzer
    assert DEFAULT_BASELINE_STRATEGY != "stacktrace-callgraph-weighted"
    result = _lens().diagnose(Path("."), error_text="no trace here")
    assert result.analysis_method == "baseline-stacktrace-callgraph-weighted"
    assert result.model_id is None


def test_constants_are_consistent() -> None:
    n = len(constants.FEATURES)
    assert n == len(constants.MEAN) == len(constants.SCALE) == len(constants.COEFFICIENTS) == 15
    assert all(scale > 0 for scale in constants.SCALE)


def test_test_files_are_ranked_after_production_code(repo: Path) -> None:
    result = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    assert result.status is AnalysisStatus.OK
    files = [c.file for c in result.candidates]
    assert files[0] == "service.py"
    assert files[-1] == "tests/test_service.py"
    assert all(c.score_type is ScoreType.HEURISTIC for c in result.candidates)
    test_candidate = result.candidates[-1]
    assert any(e.kind is EvidenceKind.PATH_MATCH for e in test_candidate.evidence)


def test_is_deterministic(repo: Path) -> None:
    first = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    second = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    assert [(c.file, c.function, c.score) for c in first.candidates] == [
        (c.file, c.function, c.score) for c in second.candidates
    ]


def test_no_resolvable_frames_is_not_guessed(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(SERVICE_PY, encoding="utf-8")
    log = 'Traceback (most recent call last):\n  File "/elsewhere/x.py", line 1, in f\n    g()\nKeyError: 1\n'
    result = _lens().diagnose(tmp_path, error_text=log)
    assert result.status is AnalysisStatus.NO_CANDIDATES
    assert result.candidates == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_a.py", True),
        ("pkg/tests/helpers.py", True),
        ("pkg/test_x.py", True),
        ("pkg/x_test.py", True),
        ("conftest.py", True),
        ("pkg/testing/util.py", True),
        ("pkg/core.py", False),
        ("pkg/contest.py", False),
        ("src/latest/x.py", False),
    ],
)
def test_is_test_file(path: str, expected: bool) -> None:
    assert is_test_file(path) is expected
