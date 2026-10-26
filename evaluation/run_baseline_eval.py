#!/usr/bin/env python
"""Baseline evaluation harness for FailureLens.

Runs the dependency-light baseline analyzer over a labelled benchmark file and
reports *real* ranking metrics (MRR, top-k accuracy, recall@k) at both the file
and function level. No numbers are hard-coded: every value is computed from the
analyzer's actual output on the referenced repositories and logs.

Usage::

    python evaluation/run_baseline_eval.py
    python evaluation/run_baseline_eval.py --benchmark evaluation/benchmarks/sample_cases.json
    python evaluation/run_baseline_eval.py --output evaluation/reports/baseline.json

The harness only *measures* the baseline; it never modifies the target
repositories and performs no network access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from failurelens import FailureLens
from failurelens.evaluation import RankingCase, evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = REPO_ROOT / "evaluation" / "benchmarks" / "sample_cases.json"


def _unique(items: list[str]) -> list[str]:
    """De-duplicate while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _diagnose_case(lens: FailureLens, case: dict[str, Any]) -> tuple[RankingCase, RankingCase]:
    """Return (file-level case, function-level case) for one benchmark entry."""
    repo = REPO_ROOT / case["repository"]
    log = REPO_ROOT / case["error_log"]
    result = lens.diagnose(repository=str(repo), error_log=str(log), top_k=None)

    predicted_files = _unique([c.file for c in result.candidates if c.file])
    predicted_functions = _unique(
        [f"{c.file}::{c.function}" for c in result.candidates if c.file and c.function]
    )

    file_case = RankingCase(
        predicted=predicted_files,
        relevant=frozenset(case.get("relevant_files", [])),
        case_id=case.get("case_id"),
    )
    function_case = RankingCase(
        predicted=predicted_functions,
        relevant=frozenset(case.get("relevant_functions", [])),
        case_id=case.get("case_id"),
    )
    return file_case, function_case


def run(benchmark_path: Path) -> dict[str, Any]:
    """Evaluate the baseline against ``benchmark_path`` and return a report."""
    data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    lens = FailureLens.from_baseline()

    file_cases: list[RankingCase] = []
    function_cases: list[RankingCase] = []
    for case in cases:
        file_case, function_case = _diagnose_case(lens, case)
        file_cases.append(file_case)
        # Only score functions when the case actually provides function labels.
        if case.get("relevant_functions"):
            function_cases.append(function_case)

    report: dict[str, Any] = {
        "benchmark": str(benchmark_path.relative_to(REPO_ROOT)),
        "analysis_method": lens.analyzer.analysis_method,
        "file_level": dict(evaluate(file_cases)),
    }
    if function_cases:
        report["function_level"] = dict(evaluate(function_cases))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the FailureLens baseline.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="Path to a benchmark JSON file (see evaluation/benchmarks/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to also write the JSON report to.",
    )
    args = parser.parse_args(argv)

    if not args.benchmark.exists():
        print(f"benchmark not found: {args.benchmark}", file=sys.stderr)
        return 2

    report = run(args.benchmark)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
