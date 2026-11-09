"""Record a v0.6 reranker report in MLflow: one run per candidate plus one selection run.

Usage::

    python -m evaluation.record_reranker --report ~/.local/share/failurelens/v06-reports/v0.6-dev.json \\
        [--tracking-uri URI]

Idempotent (a candidate already recorded for the same report is skipped) and
aggregate-only: only numbers from the report are stored, never case data, and no
model files are uploaded. The reranker has no epochs, so there are no loss curves; the
tracked evidence is the dev and cross-validation scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluation.record_full_pool import REPO_ROOT, _already_recorded

from failurelens.experiment_tracking import MLflowConfig, track_experiment

KIND = "reranker-experiment"
_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-/ ]")


def _name(value: str) -> str:
    return _UNSAFE.sub("_", value)


def _put(metrics: dict[str, float], key: str, value: Any) -> None:
    if isinstance(value, (bool, int, float)):
        metrics[_name(key)] = float(value)


def candidate_metrics(report: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for field in ("function_mrr", "function_top1", "delta_mrr_vs_v5", "delta_top1_vs_v5"):
        _put(metrics, f"dev.{field}", row[field])
    interval = row.get("delta_mrr_ci95")
    if interval:
        _put(metrics, "dev.delta_mrr_ci_lower", interval[0])
        _put(metrics, "dev.delta_mrr_ci_upper", interval[1])
    for repository, value in row.get("mrr_by_repository", {}).items():
        _put(metrics, f"dev.mrr.{repository}", value)
    for field in ("train_cv_function_mrr", "train_cv_function_top1", "train_cv_delta_vs_v5"):
        _put(metrics, field.replace("train_cv_", "train_cv."), row[field])
    _put(metrics, "fit_seconds", row["fit_seconds"])
    _put(metrics, "dev.p95_seconds", row["dev_p95_seconds"])
    _put(metrics, "selected", row["method"] == f"candidate-{report['selected']}")
    _put(metrics, "train_cases", report["train_cases"])
    _put(metrics, "dev_cases", report["dev_cases"])
    return metrics


def summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, row in report["references"].items():
        if isinstance(row, dict):
            _put(metrics, f"dev.{name}.function_mrr", row["function_mrr"])
            _put(metrics, f"dev.{name}.function_top1", row["function_top1"])
    selected = report["selected_summary"]
    _put(metrics, "dev.selected.function_mrr", selected["function_mrr"])
    _put(metrics, "dev.selected.delta_mrr_vs_v5", selected["delta_mrr_vs_v5"])
    rules = report["rules"]
    for rule, detail in rules.items():
        if isinstance(detail, dict):
            _put(metrics, f"rule.{rule}.pass", detail["pass"])
    _put(metrics, "rule.all_pass", rules["all_pass"])
    _put(metrics, "verdict_pass", report["verdict"] == "pass")
    _put(metrics, "loro.mrr_delta_vs_v5", report["leave_one_repo_out_selection"]["mrr_delta_vs_v5"])
    ablation = report["embedding_ablation"]
    _put(metrics, "ablation.best_without_embedding_mrr", ablation["best_without_embedding_mrr"])
    _put(metrics, "ablation.best_with_embedding_mrr", ablation["best_with_embedding_mrr"])
    _put(metrics, "pool.dev_function_recall", report["pool_recall_dev"]["function_recall"])
    _put(metrics, "resources.peak_rss_gib", report["resources"]["peak_rss_gib"])
    return metrics


def _record_one(
    *,
    digest: str,
    run_name: str,
    version: str,
    params: dict[str, str],
    metrics: dict[str, float],
    payload: dict[str, Any],
    tracking_uri: str | None,
) -> str:
    config = MLflowConfig(tracking_uri=tracking_uri, run_name=run_name, model_version=version)
    if _already_recorded(config, digest):
        return f"{run_name}: skipped (already recorded)"
    with track_experiment(config, kind=KIND, repository=REPO_ROOT) as experiment:
        experiment.log_params({"report_sha256": digest, **params})
        experiment.log_metrics(metrics)
        experiment.log_dict(payload, "aggregate_report.json")
    return f"{run_name}: recorded"


def record(report_path: Path, tracking_uri: str | None) -> list[str]:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    base = hashlib.sha256(raw).hexdigest()
    outcomes: list[str] = []
    for row in report["candidates"]:
        key = str(row["key"])
        digest = hashlib.sha256(f"{base}:{key}".encode()).hexdigest()
        params = {
            "label": str(report["label"]),
            "family": str(row["family"]),
            "feature_groups": ",".join(row["feature_groups"]),
            "c": str(row["c"]),
            "epochs": "not applicable (no iterative epochs)",
            "train_cases": str(report["train_cases"]),
        }
        outcomes.append(
            _record_one(
                digest=digest,
                run_name=f"v0.6-candidate-{key}",
                version=f"v0.6.0-candidate-{key.lower()}",
                params=params,
                metrics=candidate_metrics(report, row),
                payload={k: v for k, v in row.items() if k != "artifact"},
                tracking_uri=tracking_uri,
            )
        )
    digest = hashlib.sha256(f"{base}:selection".encode()).hexdigest()
    outcomes.append(
        _record_one(
            digest=digest,
            run_name="v0.6-selection",
            version="v0.6.0-selection",
            params={
                "label": str(report["label"]),
                "selected": str(report["selected"]),
                "verdict": str(report["verdict"]),
            },
            metrics=summary_metrics(report),
            payload={
                k: report[k]
                for k in ("references", "selected", "rules", "verdict", "embedding_ablation")
            },
            tracking_uri=tracking_uri,
        )
    )
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tracking-uri")
    args = parser.parse_args(argv)
    for line in record(args.report, args.tracking_uri):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
