"""Regression guard for datasets/validation/check_splits.py.

The script is the offline gate for split leakage and per-record quality. These
tests run it against the committed fixture so the gate cannot silently drift.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "datasets" / "validation" / "check_splits.py"
FIXTURE = REPO_ROOT / "datasets" / "fixtures" / "sample_record.json"


@pytest.fixture()
def checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_splits", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_alone_is_clean(checker: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    code = checker.main(["--split", f"train={FIXTURE}"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["ok"] is True
    assert report["splits"]["train"]["passed"] == 1


def test_same_repository_in_two_splits_fails(
    checker: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "copy.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    code = checker.main(["--split", f"train={FIXTURE}", "--split", f"eval={eval_dir}"])
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["ok"] is False
    assert [leak["kind"] for leak in report["leaks"]] == ["repository", "record-id"]


def test_bad_split_argument_is_usage_error(checker: ModuleType) -> None:
    with pytest.raises(SystemExit) as excinfo:
        checker.main(["--split", "train"])
    assert excinfo.value.code == 2
