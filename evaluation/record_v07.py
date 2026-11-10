"""Record a v0.7 report in MLflow: one run per candidate plus one selection run (idempotent).

Usage::

    python -m evaluation.record_v07 --report ~/.local/share/failurelens/v07-reports/v0.7-dev.json \\
        [--weighted-report .../v0.7-weighted-dev.json] [--tracking-uri URI]

Aggregate-only: only numbers from the reports are stored. The candidates have no epochs
(rule lists and one-shot fits), so there are no loss curves; the tracked evidence is the
development and leave-one-repository-out cross-validation scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation.record_reranker import _put, _record_one


def candidate_metrics(report: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in ("function_mrr", "function_top1", "delta_mrr_vs_v5", "delta_top1_vs_v5"):
        _put(metrics, f"dev.{field}", row[field])
    for name, interval in (
        ("mrr", row.get("delta_mrr_ci95")),
        ("top1", row.get("delta_top1_ci95")),
    ):
        if interval:
            _put(metrics, f"dev.delta_{name}_ci_lower", interval[0])
            _put(metrics, f"dev.delta_{name}_ci_upper", interval[1])
    for kind in ("mrr_by_repository", "top1_by_repository"):
        for repository, value in row.get(kind, {}).items():
            _put(metrics, f"dev.{kind.split('_')[0]}.{repository}", value)
    for field in (
        "train_cv_function_mrr",
        "train_cv_function_top1",
        "train_cv_delta_mrr_vs_v5",
    ):
        _put(metrics, field.replace("train_cv_", "train_cv."), row[field])
    _put(metrics, "fit_seconds", row["fit_seconds"])
    _put(metrics, "dev.p95_seconds", row["dev_p95_seconds"])
    _put(metrics, "selected", row["method"] == f"candidate-{report['selected']}")
    _put(metrics, "adoptable", row["adoptable"])
    _put(metrics, "train_cases", report["train_cases"])
    _put(metrics, "dev_cases", report["dev_cases"])
    return metrics


def summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    v5 = report["references"]["V5"]
    _put(metrics, "dev.V5.function_mrr", v5["function_mrr"])
    _put(metrics, "dev.V5.function_top1", v5["function_top1"])
    selected = report["selected_summary"]
    for field in ("function_mrr", "function_top1", "delta_mrr_vs_v5", "delta_top1_vs_v5"):
        _put(metrics, f"dev.selected.{field}", selected[field])
    for rule, detail in report["rules"].items():
        if isinstance(detail, dict):
            _put(metrics, f"rule.{rule}.pass", detail["pass"])
    _put(metrics, "rule.all_pass", report["rules"]["all_pass"])
    _put(metrics, "verdict_pass", report["verdict"] == "pass")
    _put(metrics, "resources.peak_rss_gib", report["resources"]["peak_rss_gib"])
    _put(metrics, "pool.dev_function_recall", report["pool_recall_dev"]["function_recall"])
    return metrics


def weighted_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for side in ("before", "after"):
        _put(metrics, f"dev.{side}.function_mrr", report[side]["function_mrr"])
        _put(metrics, f"dev.{side}.function_top1", report[side]["function_top1"])
    for name in ("mrr_delta", "top1_delta"):
        _put(metrics, f"dev.{name}", report[name]["delta"])
        _put(metrics, f"dev.{name}_ci_lower", report[name]["ci95"][0])
    parity = report["parity_with_evaluator_candidate_d"]
    _put(metrics, "parity.identical_full_order_cases", parity["identical_full_order_cases"])
    _put(metrics, "parity.cases", parity["cases"])
    _put(
        metrics,
        "performance.p95_seconds_including_index",
        report["timing_seconds"]["after_index_plus_diagnose_p95"],
    )
    _put(metrics, "performance.peak_rss_gib", report["peak_rss_gib"])
    return metrics


def record(report_path: Path, weighted_path: Path | None, tracking_uri: str | None) -> list[str]:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    base = hashlib.sha256(raw).hexdigest()
    outcomes: list[str] = []
    for row in report["candidates"]:
        key = str(row["key"])
        digest = hashlib.sha256(f"{base}:{key}".encode()).hexdigest()
        params = {
            "label": str(report["label"]),
            "description": str(row["description"]),
            "epochs": "not applicable (no iterative epochs)",
            "train_cases": str(report["train_cases"]),
        }
        outcomes.append(
            _record_one(
                digest=digest,
                run_name=f"v0.7-candidate-{key}",
                version=f"v0.7.0-candidate-{key.lower()}",
                params=params,
                metrics=candidate_metrics(report, row),
                payload=row,
                tracking_uri=tracking_uri,
            )
        )
    digest = hashlib.sha256(f"{base}:selection".encode()).hexdigest()
    outcomes.append(
        _record_one(
            digest=digest,
            run_name="v0.7-selection",
            version="v0.7.0-selection",
            params={
                "label": str(report["label"]),
                "selected": str(report["selected"]),
                "verdict": str(report["verdict"]),
            },
            metrics=summary_metrics(report),
            payload={k: report[k] for k in ("references", "selected", "rules", "verdict")},
            tracking_uri=tracking_uri,
        )
    )
    if weighted_path is not None:
        wraw = weighted_path.read_bytes()
        wreport = json.loads(wraw)
        outcomes.append(
            _record_one(
                digest=hashlib.sha256(wraw).hexdigest(),
                run_name="v0.7-weighted-library-check",
                version="v0.7.0-weighted-strategy",
                params={
                    "label": str(wreport["label"]),
                    "weights_sha256": str(wreport["weights_sha256"]),
                    "epochs": "not applicable (no iterative epochs)",
                },
                metrics=weighted_metrics(wreport),
                payload=wreport,
                tracking_uri=tracking_uri,
            )
        )
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weighted-report", type=Path)
    parser.add_argument("--tracking-uri")
    args = parser.parse_args(argv)
    for line in record(args.report, args.weighted_report, args.tracking_uri):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
