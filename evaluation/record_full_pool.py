"""Record v0.4 full-pool aggregate reports in MLflow (idempotent, aggregate-only).

Usage::

    python -m evaluation.record_full_pool --report REPORT.json --run-name NAME \\
        --model-version v0.4.0-dev [--tracking-uri URI]

A report already recorded (same SHA-256) is skipped, so re-running never duplicates
runs. Only the numbers in the aggregate report are stored; no case data exists in it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from failurelens.experiment_tracking import MLflowConfig, track_experiment

REPO_ROOT = Path(__file__).resolve().parent.parent
KIND = "full-pool-evaluation"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-/ ]")


def _name(value: str) -> str:
    return _UNSAFE.sub("_", value)


def flatten(report: dict[str, Any]) -> dict[str, float]:
    """Numeric aggregates of a dev or final report as MLflow metrics."""
    metrics: dict[str, float] = {}

    def put(key: str, value: Any) -> None:
        if isinstance(value, (bool, int, float)):
            metrics[_name(key)] = float(value)

    for row in report.get("single_methods", []):
        for field in ("function_mrr", "function_top1", "delta_mrr_vs_L1"):
            put(f"dev.{row['method']}.{field}", row[field])
    for row in report.get("best_fused_per_model", []):
        for field in ("function_mrr", "function_top1", "delta_mrr_vs_L1"):
            put(f"dev.best_fused.{row['method'].split('|')[0]}.{field}", row[field])
    if "stop_rule" in report:
        selected = report["selected"]
        put("dev.selected.function_mrr", selected["function_mrr"])
        put("dev.selected.function_top1", selected["function_top1"])
        put(
            "dev.stop_rule.mrr_delta_vs_lexical_full",
            report["stop_rule"]["mrr_delta_vs_lexical_full"],
        )
        put("dev.stop_rule.ci_lower", report["stop_rule"]["ci95"][0])
        put("dev.stop_rule.pass", report["stop_rule"]["pass"])
        put("dev.leave_one_repo_out.function_mrr", report["leave_one_repo_out"]["function_mrr"])
        put("dev.leave_one_repo_out.pass", report["leave_one_repo_out"]["pass"])
        put("dev.configurations_searched", report["configurations_searched"])
    elif "threshold_checks" in report:
        put("final.function_mrr", report["selected"]["function_mrr"])
        put("final.function_top1", report["selected"]["function_top1"])
        for reference, row in report["comparisons"].items():
            for field in ("reference_mrr", "reference_top1", "mrr_delta", "top1_delta"):
                put(f"final.vs_{reference}.{field}", row[field])
        for check, passed in report["threshold_checks"].items():
            put(f"final.check.{check}", passed)
        put("final.latency_p95_s", report["timing"]["index_plus_rank"]["p95_s"])
        put("final.peak_rss_gib", report["resources"]["peak_rss_gib"])
    put("pool.function_recall", report["pool_recall"]["function_recall"])
    put("pool.size_median", report["pool_recall"]["pool_size_median"])
    put("pool.size_max", report["pool_recall"]["pool_size_max"])
    put("cases", report["cases"])
    put("repositories", report["repositories"])
    if "timing" in report and "embedding_per_case" in report["timing"]:
        put("dev.index_p95_s", report["timing"]["index"]["p95_s"])
        put("dev.peak_rss_gib", report["resources"]["peak_rss_gib"])
    return metrics


def record(report_path: Path, run_name: str, model_version: str, tracking_uri: str | None) -> str:
    """Create one run for the report unless an identical report was already recorded."""
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
                "selected_method": str(report.get("selected_method_from_dev", ""))
                or str(report.get("selected", {}).get("method", "")),
            }
        )
        experiment.log_metrics(flatten(report))
        experiment.log_dict(report, "aggregate_report.json")
    return "recorded"


def _already_recorded(config: MLflowConfig, digest: str) -> bool:
    import importlib

    client = importlib.import_module("mlflow.tracking").MlflowClient
    uri = config.tracking_uri or f"sqlite:///{REPO_ROOT / '.mlflow' / 'mlflow.db'}"
    handle = client(tracking_uri=uri)
    experiment = handle.get_experiment_by_name(config.experiment_name or "failurelens-development")
    if experiment is None:
        return False
    runs = handle.search_runs(
        [experiment.experiment_id], filter_string=f"params.report_sha256 = '{digest}'"
    )
    return len(runs) > 0


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
