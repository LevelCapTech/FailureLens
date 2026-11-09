"""Record a v0.5 before/after aggregate report in MLflow (idempotent, aggregate-only).

Usage::

    python -m evaluation.record_callgraph_eval --report REPORT.json \\
        --run-name NAME --model-version v0.5.0-callgraph-baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation.record_full_pool import REPO_ROOT, _already_recorded

from failurelens.experiment_tracking import MLflowConfig, track_experiment

KIND = "baseline-evaluation"


def flatten(report: dict[str, Any]) -> dict[str, float]:
    """Numeric aggregates of a v0.5 report as MLflow metrics."""
    rule = report["adoption_rule"]
    first = rule["criterion_1_mrr_gain_ge_0.10_and_ci_lower_gt_0"]
    metrics: dict[str, float] = {
        "dev.before.function_mrr": report["before"]["function_mrr"],
        "dev.before.function_top1": report["before"]["function_top1"],
        "dev.after.function_mrr": report["after"]["function_mrr"],
        "dev.after.function_top1": report["after"]["function_top1"],
        "dev.mrr_delta": first["delta"],
        "dev.mrr_delta_ci_lower": first["ci95"][0],
        "dev.mrr_delta_ci_upper": first["ci95"][1],
        "dev.rule.criterion_1_pass": float(first["pass"]),
        "dev.rule.criterion_2_pass": float(
            rule["criterion_2_leave_one_repo_out_all_positive"]["pass"]
        ),
        "dev.rule.criterion_4_pass": float(rule["criterion_4_speed_and_memory"]["pass"]),
        "performance.p95_seconds_including_index": report["timing_seconds"][
            "after_index_plus_diagnose_p95"
        ],
        "performance.peak_rss_gib": report["peak_rss_gib"],
        "cases": float(report["cases"]),
        "repositories": float(report["repositories"]),
    }
    parity = report["parity_with_v0.4_selected_method"]
    metrics["parity.identical_full_order_cases"] = float(parity["identical_full_order_cases"])
    return metrics


def record(report_path: Path, run_name: str, model_version: str, tracking_uri: str | None) -> str:
    raw = report_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    config = MLflowConfig(tracking_uri=tracking_uri, run_name=run_name, model_version=model_version)
    if _already_recorded(config, digest):
        return "skipped (already recorded)"
    with track_experiment(config, kind=KIND, repository=REPO_ROOT) as experiment:
        experiment.log_params(
            {
                "report_sha256": digest,
                "label": str(report.get("label", "")),
                "pool": "full-repository",
                "analysis_method": str(report["after"]["analysis_method"]),
                "compared_with": str(report["before"]["analysis_method"]),
                "independent_evidence": "false",
            }
        )
        experiment.log_metrics(flatten(report))
        experiment.log_dict(report, "aggregate_report.json")
    return "recorded"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--tracking-uri")
    args = parser.parse_args(argv)
    print(record(args.report, args.run_name, args.model_version, args.tracking_uri))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
