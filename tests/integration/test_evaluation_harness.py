"""Integration test: the baseline evaluation harness runs on the bundled benchmark.

This exercises ``evaluation/run_baseline_eval.py`` end to end against the real
sample fixture and asserts the *honest* metrics it should produce. The baseline
over-weights the exception site, so the root-cause function (``average``) ranks
second - the test encodes that real behaviour rather than an aspirational score.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

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
