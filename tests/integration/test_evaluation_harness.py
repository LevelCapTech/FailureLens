"""Integration test: the baseline evaluation harness runs on the bundled benchmark.

This exercises ``evaluation/run_baseline_eval.py`` end to end against the real
sample fixture and asserts the *honest* metrics it should produce. The baseline
over-weights the exception site, so the root-cause function (``average``) ranks
second - the test encodes that real behaviour rather than an aspirational score.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "evaluation" / "run_baseline_eval.py"
BENCHMARK_PATH = REPO_ROOT / "evaluation" / "benchmarks" / "sample_cases.json"


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flens_eval_harness", HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_reports_real_metrics_on_fixture() -> None:
    harness = _load_harness()
    report = harness.run(BENCHMARK_PATH)

    assert report["analysis_method"] == "baseline-stacktrace-lexical"

    file_level = report["file_level"]
    assert file_level["num_cases"] == 1.0
    # The baseline reliably picks the right *file* (core.py) at rank 1.
    assert file_level["top_1_accuracy"] == 1.0
    assert file_level["mrr"] == 1.0

    function_level = report["function_level"]
    assert function_level["num_cases"] == 1.0
    # Root cause (average) ranks 2nd behind the exception site (divide):
    assert function_level["top_1_accuracy"] == 0.0
    assert function_level["mrr"] == 0.5
    assert function_level["top_5_accuracy"] == 1.0


def test_unlabelled_functions_and_empty_benchmarks_have_no_invented_measurements(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    benchmark = json.loads(BENCHMARK_PATH.read_text())
    for case in benchmark["cases"]:
        case.pop("relevant_functions", None)
    path = tmp_path / "files-only.json"
    path.write_text(json.dumps(benchmark))
    report = harness.run(path)
    assert "function_level" not in report
    assert report["performance"]["diagnosis_mean_ms"] >= 0
    path.write_text('{"cases": []}')
    empty = harness.run(path)
    assert "function_level" not in empty
    assert "performance" not in empty


def test_plain_harness_does_not_import_optional_stacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://unused.invalid")
    script = """
import importlib.abc
import runpy
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mlflow", "torch", "transformers", "sentence_transformers"}:
            raise AssertionError("Unexpected optional import: " + fullname)

sys.meta_path.insert(0, BlockOptional())
sys.argv = ["evaluation/run_baseline_eval.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout)["analysis_method"] == "baseline-stacktrace-lexical"
