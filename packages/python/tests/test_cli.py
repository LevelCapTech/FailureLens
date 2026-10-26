"""Tests for the ``failurelens`` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from failurelens import __version__
from failurelens.cli import ExitCode, app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == ExitCode.OK
    assert __version__ in result.stdout


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == ExitCode.OK
    assert __version__ in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == ExitCode.OK
    for command in ("diagnose", "index", "models", "version"):
        assert command in result.stdout


def _write_log(tmp_path: Path, traceback: str) -> Path:
    log = tmp_path / "failure.log"
    log.write_text(traceback, encoding="utf-8")
    return log


def test_diagnose_text_output(sample_repo: Path, sample_traceback: str, tmp_path: Path) -> None:
    log = _write_log(tmp_path, sample_traceback)
    result = runner.invoke(
        app, ["diagnose", "-r", str(sample_repo), "-l", str(log), "-k", "3", "--quiet"]
    )
    assert result.exit_code == ExitCode.OK
    assert "divide" in result.stdout
    assert "ZeroDivisionError" in result.stdout


def test_diagnose_json_output_is_clean(
    sample_repo: Path, sample_traceback: str, tmp_path: Path
) -> None:
    log = _write_log(tmp_path, sample_traceback)
    result = runner.invoke(
        app,
        ["diagnose", "-r", str(sample_repo), "-l", str(log), "-f", "json", "--quiet"],
    )
    assert result.exit_code == ExitCode.OK
    payload = json.loads(result.stdout)  # stdout must be valid JSON on its own
    assert payload["status"] == "ok"
    assert payload["schema_version"]
    assert payload["candidates"][0]["function"] == "divide"


def test_diagnose_missing_log_returns_input_error(sample_repo: Path) -> None:
    result = runner.invoke(
        app, ["diagnose", "-r", str(sample_repo), "-l", "/no/such/file.log", "--quiet"]
    )
    assert result.exit_code == ExitCode.INPUT_ERROR


def test_diagnose_pretrained_without_ml_returns_dependency_error(
    sample_repo: Path, sample_traceback: str, tmp_path: Path
) -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        log = _write_log(tmp_path, sample_traceback)
        result = runner.invoke(
            app,
            ["diagnose", "-r", str(sample_repo), "-l", str(log), "-m", "some/model", "--quiet"],
        )
        assert result.exit_code == ExitCode.DEPENDENCY
    else:  # pragma: no cover - only when the ml extra is installed
        import pytest

        pytest.skip("ml extra installed; dependency guard not exercised")


def test_diagnose_no_candidates_exit_code(sample_repo: Path, tmp_path: Path) -> None:
    text = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.11/json/decoder.py", line 1, in decode\n'
        "    raise ValueError\n"
        "ValueError: bad\n"
    )
    log = _write_log(tmp_path, text)
    result = runner.invoke(app, ["diagnose", "-r", str(sample_repo), "-l", str(log), "--quiet"])
    assert result.exit_code == ExitCode.INCONCLUSIVE


def test_models_list_and_info() -> None:
    listing = runner.invoke(app, ["models", "list"])
    assert listing.exit_code == ExitCode.OK
    assert "baseline" in listing.stdout

    info = runner.invoke(app, ["models", "info", "-m", "baseline"])
    assert info.exit_code == ExitCode.OK
    assert "baseline" in info.stdout


def test_index_command_reports_counts(sample_repo: Path) -> None:
    result = runner.invoke(app, ["index", "-r", str(sample_repo), "--quiet"])
    assert result.exit_code == ExitCode.OK
    assert "file(s)" in result.stdout
