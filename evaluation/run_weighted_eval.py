#!/usr/bin/env python
"""Aggregate-only evaluation of the library's ``stacktrace-callgraph-weighted`` strategy.

Compares the library analyzers ``stacktrace-callgraph`` (before, v0.5 default) and
``stacktrace-callgraph-weighted`` (after, the frozen v0.7 candidate D) on the
whole-repository pool of verified failing trees, with the v0.4 convention (library
ranking first, remaining pool functions by candidate ID). It also checks that the
library reproduces the evaluator-side candidate D (frozen constants) case by case.
Nothing is trained; no per-case data is printed or stored.

Development check (v0.3 dev records) or, for the one-time final measurement, a new
evaluation set whose use the owner has approved::

    python -m evaluation.run_weighted_eval --records R.json --records-sha256 <hex> \\
      --source-manifest S.json --train-manifest T.json --tree-manifest F.json --dev
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from evaluation.full_pool_methods import order_by
from evaluation.independent_runner import _percentile, _rss_gib, paired_bootstrap
from evaluation.private_adapter import PrivateInputError
from evaluation.product_pool import load_product_cases, verify_failing_tree
from evaluation.reranker_features import case_features
from evaluation.run_callgraph_eval import candidate_id, extend_with_pool
from evaluation.run_full_pool import Metric, _mean, _round, per_case_metrics, write_report
from evaluation.run_reranker import per_repository_mrr
from evaluation.v07_features import case_v07, learned_matrix

from failurelens import FailureLens
from failurelens.inference import weighted_constants as constants

DEFAULT_OUT = Path.home() / ".local" / "share" / "failurelens" / "v07-reports"


def frozen_order(request: Any, parse: Any) -> tuple[str, ...]:
    """Evaluator-side order of the frozen candidate D (constants, no fitting)."""
    f = case_features(request, parse)
    c = case_v07(request, parse)
    matrix = (learned_matrix(f, c) - np.array(constants.MEAN)) / np.array(constants.SCALE)
    scores = matrix @ np.array(constants.COEFFICIENTS) + constants.INTERCEPT
    return order_by(f.ids, [float(v) for v in scores])


def _delta(
    strata: Sequence[str], before: Sequence[float], after: Sequence[float]
) -> dict[str, Any]:
    interval = paired_bootstrap(strata, before, after)["delta"]
    return {
        "delta": round(_mean(after) - _mean(before), 4),
        "ci95": _round(interval),
        "ci_lower_gt_0": bool(interval is not None and interval[0] > 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--tree-manifest", type=Path, required=True)
    parser.add_argument("--dev", action="store_true", help="relax the 30-case/3-repo floor")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label", default="v0.7-weighted-dev")
    args = parser.parse_args(argv)

    cases, _ = load_product_cases(
        args.records,
        args.records_sha256,
        args.source_manifest,
        args.train_manifest,
        args.tree_manifest,
        top_k=5,
        production=not args.dev,
        allow_ambiguous_retrieved_names=True,
        pool="full",
    )
    payload = json.loads(args.records.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    entries = json.loads(args.tree_manifest.read_text(encoding="utf-8"))["cases"]
    if len(records) != len(cases):
        raise PrivateInputError("weighted evaluation rejected: record/case mismatch")

    before_lens = FailureLens.from_baseline(strategy="stacktrace-callgraph")
    after_lens = FailureLens.from_baseline(strategy="stacktrace-callgraph-weighted")
    strata = [case.repository_id for case in cases]
    before_orders: list[tuple[str, ...]] = []
    after_orders: list[tuple[str, ...]] = []
    seconds: list[float] = []
    parity_full = 0
    parity_top5 = 0
    for record, case in zip(records, cases, strict=True):
        root = verify_failing_tree(entries[record["record_id"]], record["source_commit"])
        pool_ids = [c.id for c in case.request.candidates]

        index = before_lens.index(root)
        result = before_lens.diagnose(
            root, error_text=record["error_log"], top_k=len(pool_ids), index=index
        )
        before_orders.append(
            extend_with_pool([candidate_id(c) for c in result.candidates], pool_ids)
        )

        began = time.perf_counter()
        index = after_lens.index(root)
        result = after_lens.diagnose(
            root, error_text=record["error_log"], top_k=len(pool_ids), index=index
        )
        seconds.append(time.perf_counter() - began)
        ordered = extend_with_pool([candidate_id(c) for c in result.candidates], pool_ids)
        after_orders.append(ordered)

        reference = frozen_order(case.request, after_lens.parsers.parse)
        parity_full += int(reference == ordered)
        parity_top5 += int(reference[:5] == ordered[:5])

    before: Metric = per_case_metrics(cases, before_orders)
    after: Metric = per_case_metrics(cases, after_orders)
    p95 = _percentile(seconds, 0.95)
    report: dict[str, Any] = {
        "label": args.label,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": len(cases),
        "repositories": len(set(strata)),
        "pool": "full-repository (all functions and methods of the failing tree)",
        "weights_sha256": constants.WEIGHTS_SHA256,
        "before": {
            "analysis_method": before_lens.analyzer.analysis_method,
            "function_mrr": round(_mean(before[0]), 4),
            "function_top1": round(_mean(before[1]), 4),
            "mrr_by_repository": per_repository_mrr(strata, before[0]),
            "top1_by_repository": per_repository_mrr(strata, before[1]),
        },
        "after": {
            "analysis_method": after_lens.analyzer.analysis_method,
            "function_mrr": round(_mean(after[0]), 4),
            "function_top1": round(_mean(after[1]), 4),
            "mrr_by_repository": per_repository_mrr(strata, after[0]),
            "top1_by_repository": per_repository_mrr(strata, after[1]),
        },
        "mrr_delta": _delta(strata, before[0], after[0]),
        "top1_delta": _delta(strata, before[1], after[1]),
        "parity_with_evaluator_candidate_d": {
            "identical_full_order_cases": parity_full,
            "identical_top5_cases": parity_top5,
            "cases": len(cases),
        },
        "timing_seconds": {
            "after_index_plus_diagnose_p95": round(p95, 3),
            "after_index_plus_diagnose_mean": round(sum(seconds) / len(seconds), 3),
        },
        "peak_rss_gib": round(_rss_gib(), 3),
    }
    path = write_report(report, args.out_dir, f"{args.label}.json")
    print(json.dumps(report, indent=2))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
