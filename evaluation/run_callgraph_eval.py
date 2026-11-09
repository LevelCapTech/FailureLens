#!/usr/bin/env python
"""v0.5 before/after evaluation of the library's call-graph baseline (aggregate-only).

Compares the *library* analyzers on the whole-repository pool of verified failing
trees: ``stacktrace-lexical`` (before, as ``S0`` with the v0.4 convention: its
returned candidates first, then the rest by candidate ID) and
``stacktrace-callgraph`` (after; its ranked output, then the rest by candidate
ID). Applies the adoption rule fixed in ``docs/releases/v0.5/PLAN.md``. Nothing is
trained and no per-case data is printed or stored. Reports are owner-only files.

Usage (dev set; no model needed)::

    python -m evaluation.run_callgraph_eval --records R.json --records-sha256 <hex> \\
      --source-manifest S.json --train-manifest T.json --tree-manifest F.json --dev
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.full_pool_methods import build_components, fused_scores, order_by
from evaluation.independent_runner import _percentile, _rss_gib, paired_bootstrap
from evaluation.private_adapter import PrivateInputError
from evaluation.product_pool import load_product_cases, verify_failing_tree
from evaluation.run_full_pool import Metric, _mean, _round, per_case_metrics, write_report

from failurelens import FailureLens
from failurelens.schemas import Candidate

DEFAULT_OUT = Path.home() / ".local" / "share" / "failurelens" / "v05-reports"
MIN_MRR_GAIN = 0.10
MAX_P95_SECONDS = 5.0
MAX_RSS_GIB = 8.0
V04_WEIGHTS = (0.0, 0.0, 0.25, 0.25)


def candidate_id(candidate: Candidate) -> str | None:
    """Pool-style ID of a function-level candidate (``None`` for file-level ones)."""
    if candidate.function is None or candidate.start_line is None or candidate.end_line is None:
        return None
    return f"{candidate.file}::{candidate.function}:{candidate.start_line}-{candidate.end_line}"


def extend_with_pool(ranked: Sequence[str | None], pool_ids: Sequence[str]) -> tuple[str, ...]:
    """Library order restricted to the pool, then the remaining pool ids ascending."""
    members = set(pool_ids)
    head: list[str] = []
    seen: set[str] = set()
    for identifier in ranked:
        if identifier is not None and identifier in members and identifier not in seen:
            head.append(identifier)
            seen.add(identifier)
    return tuple(head) + tuple(sorted(item for item in pool_ids if item not in seen))


def _delta(
    strata: Sequence[str], before: Sequence[float], after: Sequence[float]
) -> dict[str, Any]:
    boot = paired_bootstrap(strata, before, after)["delta"]
    return {
        "delta": round(_mean(after) - _mean(before), 4),
        "ci95": _round(boot),
        "ci_lower_gt_0": bool(boot is not None and boot[0] > 0),
    }


def rule_checks(
    strata: Sequence[str], before: Metric, after: Metric, p95_seconds: float, rss_gib: float
) -> dict[str, Any]:
    """Adoption criteria 1, 2 and 4 of docs/releases/v0.5/PLAN.md, applied literally."""
    main = _delta(strata, before[0], after[0])
    main["pass"] = bool(main["delta"] >= MIN_MRR_GAIN and main["ci_lower_gt_0"])
    per_repo: dict[str, Any] = {}
    leave_out_pass = True
    for repo in sorted(set(strata)):
        keep = [i for i, s in enumerate(strata) if s != repo]
        sub = _delta(
            [strata[i] for i in keep], [before[0][i] for i in keep], [after[0][i] for i in keep]
        )
        sub["pass"] = bool(sub["delta"] > 0 and sub["ci_lower_gt_0"])
        leave_out_pass = leave_out_pass and sub["pass"]
        own = [i for i, s in enumerate(strata) if s == repo]
        per_repo[repo] = {
            "cases": len(own),
            "mrr_before": round(_mean([before[0][i] for i in own]), 4),
            "mrr_after": round(_mean([after[0][i] for i in own]), 4),
            "leave_this_repo_out": sub,
        }
    return {
        "criterion_1_mrr_gain_ge_0.10_and_ci_lower_gt_0": main,
        "criterion_2_leave_one_repo_out_all_positive": {
            "pass": leave_out_pass,
            "per_repository": per_repo,
        },
        "criterion_4_speed_and_memory": {
            "p95_seconds_including_index": round(p95_seconds, 3),
            "peak_rss_gib": round(rss_gib, 3),
            "pass": bool(p95_seconds <= MAX_P95_SECONDS and rss_gib <= MAX_RSS_GIB),
        },
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
    parser.add_argument("--label", default="v0.5-dev")
    args = parser.parse_args(argv)

    cases, before_lookup = load_product_cases(
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
        raise PrivateInputError("callgraph evaluation rejected: record/case mismatch")

    old = FailureLens.from_baseline(strategy="stacktrace-lexical")
    new = FailureLens.from_baseline(strategy="stacktrace-callgraph")
    strata = [case.repository_id for case in cases]
    before_orders: list[tuple[str, ...]] = []
    after_orders: list[tuple[str, ...]] = []
    new_seconds: list[float] = []
    old_seconds: list[float] = []
    parity_full = 0
    parity_ranked_prefix = 0
    for record, case in zip(records, cases, strict=True):
        root = verify_failing_tree(entries[record["record_id"]], record["source_commit"])
        pool_ids = [c.id for c in case.request.candidates]
        before_orders.append(before_lookup(case.request).ordered_ids)

        began = time.perf_counter()
        index = new.index(root)
        result = new.diagnose(
            root, error_text=record["error_log"], top_k=len(pool_ids), index=index
        )
        new_seconds.append(time.perf_counter() - began)
        ordered = extend_with_pool([candidate_id(c) for c in result.candidates], pool_ids)
        after_orders.append(ordered)

        began = time.perf_counter()
        old_index = old.index(root)
        old.diagnose(root, error_text=record["error_log"], top_k=5, index=old_index)
        old_seconds.append(time.perf_counter() - began)

        # Faithfulness to the evaluated v0.4 method (weights 0/0/0.25/0.25).
        components = build_components(case.request, new.parsers.parse)
        reference = order_by(
            components.ids, fused_scores(components, [0.0] * len(components.ids), V04_WEIGHTS)
        )
        parity_full += int(reference == ordered)
        positive = sum(
            1 for s, g in zip(components.stack, components.graph, strict=True) if s > 0 or g > 0
        )
        parity_ranked_prefix += int(reference[:positive] == ordered[:positive])

    before: Metric = per_case_metrics(cases, before_orders)
    after: Metric = per_case_metrics(cases, after_orders)
    p95 = _percentile(new_seconds, 0.95)
    checks = rule_checks(strata, before, after, p95, _rss_gib())
    report: dict[str, Any] = {
        "label": args.label,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": len(cases),
        "repositories": len(set(strata)),
        "pool": "full-repository (all functions and methods of the failing tree)",
        "before": {
            "analysis_method": old.analyzer.analysis_method,
            "function_mrr": round(_mean(before[0]), 4),
            "function_top1": round(_mean(before[1]), 4),
        },
        "after": {
            "analysis_method": new.analyzer.analysis_method,
            "function_mrr": round(_mean(after[0]), 4),
            "function_top1": round(_mean(after[1]), 4),
        },
        "top1_delta": _delta(strata, before[1], after[1]),
        "adoption_rule": checks,
        "parity_with_v0.4_selected_method": {
            "identical_full_order_cases": parity_full,
            "identical_scored_prefix_cases": parity_ranked_prefix,
            "cases": len(cases),
        },
        "timing_seconds": {
            "after_index_plus_diagnose_p95": round(p95, 3),
            "after_median": round(statistics.median(new_seconds), 3),
            "after_max": round(max(new_seconds), 3),
            "before_index_plus_diagnose_p95": round(_percentile(old_seconds, 0.95), 3),
        },
        "peak_rss_gib": round(_rss_gib(), 3),
        "selection_bias_note": (
            "Weights were selected on this development set in v0.4; this is not independent "
            "evidence."
        ),
    }
    path = write_report(report, args.out_dir, f"{args.label}.json")
    print(json.dumps({k: v for k, v in report.items() if k != "adoption_rule"}, indent=2))
    print(json.dumps(report["adoption_rule"], indent=2))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
