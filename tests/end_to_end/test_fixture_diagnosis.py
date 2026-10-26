"""End-to-end test driving the shipped example fixture.

This exercises the whole pipeline (log parsing -> indexing -> retrieval ->
ranking -> reporting) against the real ``examples/fixtures/sample_project`` and
its captured traceback, through both the Python API and the CLI, to guard
against regressions in the user-facing behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from failurelens import FailureLens
from failurelens.cli import ExitCode, app
from failurelens.schemas import AnalysisStatus

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "examples" / "fixtures" / "sample_project"
FIXTURE_LOG = FIXTURE / "failure.log"


def test_fixture_exists() -> None:
    assert FIXTURE.is_dir()
    assert FIXTURE_LOG.is_file()


def test_api_diagnoses_fixture() -> None:
    lens = FailureLens.from_baseline()
    result = lens.diagnose(FIXTURE, error_log=FIXTURE_LOG, top_k=5)

    assert result.status is AnalysisStatus.OK
    assert result.error_type == "ZeroDivisionError"
    # The exception surfaces in divide(); it should be the top-ranked candidate.
    assert result.candidates[0].file == "calculator/core.py"
    assert result.candidates[0].function == "divide"
    # The actual root cause, average(), must also appear among the candidates.
    assert any(c.function == "average" for c in result.candidates)


def test_cli_diagnoses_fixture_json() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "diagnose",
            "-r",
            str(FIXTURE),
            "-l",
            str(FIXTURE_LOG),
            "-f",
            "json",
            "--quiet",
        ],
    )
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["candidates"][0]["function"] == "divide"
