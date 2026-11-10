#!/usr/bin/env python
"""v0.7 model-free experiment (evaluator-only, aggregate-only output).

Six candidates fixed in ``docs/releases/v0.7/selection-rule.md`` before any result. Trains
on the v0.3 TRAIN records, selects on the v0.3 DEV records and applies the pre-registered
rules R1-R5 of ``docs/releases/v0.7/PLAN.md``. Reads no Gate C data and prints no per-case
data. Reports are owner-only local files. Needs scikit-learn (experiment environment)::

    python -m evaluation.run_v07 run --label v0.7-dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from evaluation.full_pool_methods import order_by
from evaluation.independent_runner import _rss_gib, paired_bootstrap
from evaluation.reranker_features import case_features
from evaluation.reranker_models import SAMPLE_SEED_BASE, TrainCase, training_subset
from evaluation.run_full_pool import (
    Metric,
    _mean,
    _round,
    per_case_metrics,
    pool_recall,
    write_report,
)
from evaluation.run_reranker import (
    DEFAULT_CORPUS,
    V5_DEV_MRR_IN_V05_REPORT,
    _p95,
    load_split,
    per_repository_mrr,
    positive_indices,
    split_paths,
    v5_order,
)
from evaluation.v07_features import CaseV07, case_v07, learned_matrix, rule_scores

from failurelens import FailureLens

DEFAULT_OUT = Path.home() / ".local" / "share" / "failurelens" / "v07-reports"
MIN_TOP1_GAIN = 0.05
MIN_MRR_GAIN = 0.03
MAX_REPO_REGRESSION = 0.05
MAX_P95_SECONDS = 5.0
MAX_RSS_GIB = 8.0
RULE_KEYS = ("A", "B", "C")
LEARNED = {
    "D": ("logistic", 0.1),
    "E": ("boosting", None),
    "F": ("pairwise", 1.0),
}
ADOPTABLE = ("A", "B", "C", "D", "F")  # E (boosted trees) is reference only
KEYS = ("A", "B", "C", "D", "E", "F")
DESCRIPTION = {
    "A": "v0.5 order, test-file functions last",
    "B": "A + call proximity from production stack functions only",
    "C": "B + 0.15 x exception-message/name match",
    "D": "logistic regression C=0.1 on 15 features",
    "E": "boosted trees on 15 features (reference only)",
    "F": "pairwise linear C=1.0 on 15 features",
}


@dataclass
class Data:
    cases: list[Any]
    f: list[Any]
    t: list[CaseV07]
    components: list[Any]
    seconds: list[float]

    @property
    def strata(self) -> list[str]:
        return [c.repository_id for c in self.cases]


def build(cases: list[Any], lens: FailureLens) -> Data:
    from evaluation.full_pool_methods import build_components

    f: list[Any] = []
    t: list[CaseV07] = []
    comp: list[Any] = []
    seconds: list[float] = []
    for case in cases:
        began = time.perf_counter()
        f.append(case_features(case.request, lens.parsers.parse))
        t.append(case_v07(case.request, lens.parsers.parse))
        seconds.append(time.perf_counter() - began)
        comp.append(build_components(case.request, lens.parsers.parse))
    return Data(cases, f, t, comp, seconds)


class Fitted:
    def __init__(self, family: str, scaler: Any, model: Any, weights: Any) -> None:
        self.family, self.scaler, self.model, self.weights = family, scaler, model, weights

    def score(self, matrix: np.ndarray) -> np.ndarray:
        if self.family == "boosting":
            return np.asarray(self.model.decision_function(matrix))
        scaled = self.scaler.transform(matrix)
        if self.family == "pairwise":
            return np.asarray(scaled @ self.weights)
        return np.asarray(self.model.decision_function(scaled))

    def coefficients(self) -> list[float] | None:
        if self.family == "pairwise":
            return [round(float(v), 4) for v in self.weights]
        if self.family == "logistic":
            return [round(float(v), 4) for v in self.model.coef_[0]]
        return None


def fit(family: str, c: float | None, data: Data, indices: Sequence[int]) -> Fitted:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    per_case: list[tuple[np.ndarray, np.ndarray]] = []
    for i in indices:
        positives = positive_indices(data.cases[i])
        if not positives:
            continue
        probe = TrainCase(data.f[i].g1, None, positives, data.f[i].ids, i)
        subset = training_subset(probe)
        matrix = learned_matrix(data.f[i], data.t[i])[subset]
        label = np.isin(subset, np.array(positives)).astype(np.int64)
        per_case.append((matrix, label))
    stacked = np.vstack([m for m, _ in per_case])
    target = np.concatenate([lab for _, lab in per_case])
    scaler = StandardScaler().fit(stacked)
    if family == "logistic":
        assert c is not None
        model = LogisticRegression(C=c, max_iter=2000).fit(scaler.transform(stacked), target)
        return Fitted(family, scaler, model, None)
    if family == "pairwise":
        assert c is not None
        diffs = []
        for matrix, label in per_case:
            scaled = scaler.transform(matrix)
            pos, neg = scaled[label == 1], scaled[label == 0]
            if pos.size and neg.size:
                diffs.append((pos[:, None, :] - neg[None, :, :]).reshape(-1, scaled.shape[1]))
        paired = np.vstack(diffs)
        data2 = np.vstack([paired, -paired])
        signs = np.concatenate([np.ones(len(paired), np.int64), np.zeros(len(paired), np.int64)])
        model = LogisticRegression(C=c, fit_intercept=False, max_iter=2000).fit(data2, signs)
        return Fitted(family, scaler, model, model.coef_[0])
    model = HistGradientBoostingClassifier(
        max_depth=3,
        max_leaf_nodes=8,
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=1.0,
        random_state=0,
        early_stopping=False,
    ).fit(stacked, target)
    return Fitted(family, scaler, model, None)


def order_learned(model: Fitted, data: Data, i: int) -> tuple[str, ...]:
    scores = model.score(learned_matrix(data.f[i], data.t[i]))
    return order_by(data.f[i].ids, [float(v) for v in scores])


def order_rule(data: Data, i: int, key: str) -> tuple[str, ...]:
    return order_by(data.t[i].ids, [float(v) for v in rule_scores(data.t[i], key)])


def metrics_for(data: Data, orders: Sequence[Sequence[str]]) -> Metric:
    return per_case_metrics(data.cases, orders)


def cross_validate(key: str, data: Data) -> Metric:
    """Leave-one-repository-out over TRAIN (rule candidates have nothing to fit)."""
    if key in RULE_KEYS:
        return metrics_for(data, [order_rule(data, i, key) for i in range(len(data.cases))])
    family, c = LEARNED[key]
    strata = data.strata
    mrr = [0.0] * len(strata)
    top1 = [0.0] * len(strata)
    for repository in sorted(set(strata)):
        fit_idx = [i for i, s in enumerate(strata) if s != repository]
        hold = [i for i, s in enumerate(strata) if s == repository]
        model = fit(family, c, data, fit_idx)
        held = per_case_metrics(
            [data.cases[i] for i in hold], [order_learned(model, data, i) for i in hold]
        )
        for pos, i in enumerate(hold):
            mrr[i], top1[i] = held[0][pos], held[1][pos]
    return mrr, top1


def select(
    metrics: dict[str, Metric], keys: Sequence[str], indices: Sequence[int] | None = None
) -> str:
    """Highest Top-1, then MRR, then rule-based before learned, then key."""

    def rank(key: str) -> tuple[float, float, bool, str]:
        mrr, top1 = metrics[key]
        chosen = range(len(mrr)) if indices is None else indices
        return (
            -_mean([top1[i] for i in chosen]),
            -_mean([mrr[i] for i in chosen]),
            key in LEARNED,
            key,
        )

    return min(keys, key=rank)


def loro_select(
    strata: Sequence[str], metrics: dict[str, Metric], keys: Sequence[str]
) -> tuple[list[float], list[float]]:
    mrr = [0.0] * len(strata)
    top1 = [0.0] * len(strata)
    for repository in sorted(set(strata)):
        others = [i for i, s in enumerate(strata) if s != repository]
        chosen = select(metrics, keys, others)
        for i, s in enumerate(strata):
            if s == repository:
                mrr[i], top1[i] = metrics[chosen][0][i], metrics[chosen][1][i]
    return mrr, top1


def summarize(name: str, metric: Metric, strata: Sequence[str], v5: Metric) -> dict[str, Any]:
    mrr_ci = paired_bootstrap(strata, v5[0], metric[0])["delta"]
    top1_ci = paired_bootstrap(strata, v5[1], metric[1])["delta"]
    return {
        "method": name,
        "function_mrr": round(_mean(metric[0]), 4),
        "function_top1": round(_mean(metric[1]), 4),
        "delta_mrr_vs_v5": round(_mean(metric[0]) - _mean(v5[0]), 4),
        "delta_mrr_ci95": _round(mrr_ci),
        "delta_top1_vs_v5": round(_mean(metric[1]) - _mean(v5[1]), 4),
        "delta_top1_ci95": _round(top1_ci),
        "mrr_by_repository": per_repository_mrr(strata, metric[0]),
        "top1_by_repository": per_repository_mrr(strata, metric[1]),
    }


def apply_rules(
    *,
    best: Metric,
    v5: Metric,
    strata: Sequence[str],
    loro_mrr: Sequence[float],
    cv_mrr: float,
    v5_train_mrr: float,
    p95_seconds: float,
    peak_rss_gib: float,
) -> dict[str, Any]:
    interval = paired_bootstrap(strata, v5[0], best[0])["delta"]
    mrr_delta = _mean(best[0]) - _mean(v5[0])
    top1_delta = _mean(best[1]) - _mean(v5[1])
    r1 = bool(
        interval is not None
        and top1_delta >= MIN_TOP1_GAIN
        and mrr_delta >= MIN_MRR_GAIN
        and interval[0] > 0
    )
    best_by = per_repository_mrr(strata, best[0])
    v5_by = per_repository_mrr(strata, v5[0])
    worst = min(best_by[r] - v5_by[r] for r in best_by)
    r2 = bool(worst >= -MAX_REPO_REGRESSION)
    loro_delta = _mean(loro_mrr) - _mean(v5[0])
    r3 = bool(loro_delta >= MIN_MRR_GAIN)
    r4 = bool(cv_mrr - v5_train_mrr > 0)
    r5 = bool(p95_seconds <= MAX_P95_SECONDS and peak_rss_gib <= MAX_RSS_GIB)
    return {
        "R1_gain_over_v5": {
            "top1_delta": round(top1_delta, 4),
            "top1_required": MIN_TOP1_GAIN,
            "mrr_delta": round(mrr_delta, 4),
            "mrr_required": MIN_MRR_GAIN,
            "mrr_ci95": _round(interval),
            "pass": r1,
        },
        "R2_no_repo_regression": {
            "worst_repo_delta": round(worst, 4),
            "allowed": -MAX_REPO_REGRESSION,
            "by_repository_candidate": best_by,
            "by_repository_v5": v5_by,
            "pass": r2,
        },
        "R3_selection_bias_check": {
            "loro_mrr": round(_mean(loro_mrr), 4),
            "mrr_delta_vs_v5": round(loro_delta, 4),
            "required": MIN_MRR_GAIN,
            "pass": r3,
        },
        "R4_train_cross_validation": {
            "cv_mrr": round(cv_mrr, 4),
            "v5_mrr_same_cases": round(v5_train_mrr, 4),
            "delta": round(cv_mrr - v5_train_mrr, 4),
            "pass": r4,
        },
        "R5_speed_memory_offline": {
            "p95_seconds": round(p95_seconds, 3),
            "peak_rss_gib": round(peak_rss_gib, 3),
            "limits": {"p95_seconds": MAX_P95_SECONDS, "peak_rss_gib": MAX_RSS_GIB},
            "model_download_or_network": False,
            "pass": r5,
        },
        "all_pass": bool(r1 and r2 and r3 and r4 and r5),
    }


def run(corpus: Path, label: str, out_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    lens = FailureLens.from_baseline()
    train_cases, _ = load_split(corpus, "train")
    dev_cases, _ = load_split(corpus, "dev")
    train = build(train_cases, lens)
    dev = build(dev_cases, lens)
    tree_entries = json.loads(split_paths(corpus, "dev")["trees"].read_text(encoding="utf-8"))[
        "cases"
    ]
    index_seconds: list[float] = []
    for entry in tree_entries.values():
        began = time.perf_counter()
        lens.index(Path(entry["root"]))
        index_seconds.append(time.perf_counter() - began)
    strata = dev.strata
    n_dev = len(dev.cases)
    v5_dev = metrics_for(dev, [v5_order(c) for c in dev.components])
    v5_train = metrics_for(train, [v5_order(c) for c in train.components])

    dev_metrics: dict[str, Metric] = {}
    dev_seconds: dict[str, list[float]] = {}
    cv_metrics: dict[str, Metric] = {}
    fit_seconds: dict[str, float] = {}
    coefficients: dict[str, list[float] | None] = {}
    all_train = list(range(len(train.cases)))
    for key in KEYS:
        per_case: list[float] = []
        orders: list[tuple[str, ...]] = []
        if key in RULE_KEYS:
            fit_seconds[key] = 0.0
            coefficients[key] = None
            for i in range(n_dev):
                began = time.perf_counter()
                orders.append(order_rule(dev, i, key))
                per_case.append(time.perf_counter() - began)
        else:
            family, c = LEARNED[key]
            began = time.perf_counter()
            model = fit(family, c, train, all_train)
            fit_seconds[key] = time.perf_counter() - began
            coefficients[key] = model.coefficients()
            for i in range(n_dev):
                began = time.perf_counter()
                orders.append(order_learned(model, dev, i))
                per_case.append(time.perf_counter() - began)
        dev_metrics[key] = metrics_for(dev, orders)
        dev_seconds[key] = per_case
        cv_metrics[key] = cross_validate(key, train)

    best = select(dev_metrics, ADOPTABLE)
    loro_mrr, _ = loro_select(strata, dev_metrics, ADOPTABLE)

    def case_seconds(key: str) -> list[float]:
        return [index_seconds[i] + dev.seconds[i] + dev_seconds[key][i] for i in range(n_dev)]

    rules = apply_rules(
        best=dev_metrics[best],
        v5=v5_dev,
        strata=strata,
        loro_mrr=loro_mrr,
        cv_mrr=_mean(cv_metrics[best][0]),
        v5_train_mrr=_mean(v5_train[0]),
        p95_seconds=_p95(case_seconds(best)),
        peak_rss_gib=_rss_gib(),
    )
    rows = []
    for key in KEYS:
        row = summarize(f"candidate-{key}", dev_metrics[key], strata, v5_dev)
        row.update(
            {
                "key": key,
                "description": DESCRIPTION[key],
                "adoptable": key in ADOPTABLE,
                "fit_seconds": round(fit_seconds[key], 2),
                "train_cv_function_mrr": round(_mean(cv_metrics[key][0]), 4),
                "train_cv_function_top1": round(_mean(cv_metrics[key][1]), 4),
                "train_cv_delta_mrr_vs_v5": round(
                    _mean(cv_metrics[key][0]) - _mean(v5_train[0]), 4
                ),
                "dev_p95_seconds": round(_p95(case_seconds(key)), 3),
                "coefficients_standardized": coefficients[key],
            }
        )
        rows.append(row)
    test_share = float(np.mean([np.mean(t.is_test) for t in dev.t]))
    report = {
        "label": label,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_cases": len(train.cases),
        "train_repositories": len(set(train.strata)),
        "dev_cases": n_dev,
        "dev_repositories": len(set(strata)),
        "references": {
            "V5": summarize("V5", v5_dev, strata, v5_dev),
            "v5_reproduces_v05_report": round(_mean(v5_dev[0]), 4) == V5_DEV_MRR_IN_V05_REPORT,
        },
        "candidates": rows,
        "selected": best,
        "selected_summary": summarize(f"candidate-{best}", dev_metrics[best], strata, v5_dev),
        "rules": rules,
        "verdict": "pass" if rules["all_pass"] else "stop",
        "pool_recall_dev": pool_recall(dev.cases),
        "mean_test_file_share_of_pool_dev": round(test_share, 4),
        "timing": {
            "index_p95_s": round(_p95(index_seconds), 3),
            "features_p95_s": round(_p95(dev.seconds), 3),
        },
        "resources": {
            "peak_rss_gib": round(_rss_gib(), 3),
            "python": sys.version.split()[0],
            "total_seconds": round(time.perf_counter() - started, 1),
        },
        "sample_seed_base": SAMPLE_SEED_BASE,
    }
    path = write_report(report, out_dir, f"{label}.json")
    print(
        json.dumps(
            {k: report[k] for k in ("selected", "selected_summary", "rules", "verdict")}, indent=2
        )
    )
    print(f"report: {path}")
    return report


FEATURE_ORDER = (
    "f01_stack_score",
    "f02_in_stack",
    "f03_deepest_frame",
    "f04_relative_depth",
    "f05_called_by_stack",
    "f06_calls_stack",
    "f07_two_hops",
    "f08_min_hops",
    "f15_name_in_log",
    "f16_same_file_as_stack",
    "t01_is_test_file",
    "t02_production_stack",
    "t03_production_hops",
    "t06_name_in_message",
    "t07_same_class_near",
)


def freeze(corpus: Path, target: Path) -> str:
    """Refit candidate D on all TRAIN records and write its constants as a Python module."""
    import hashlib

    lens = FailureLens.from_baseline()
    train_cases, _ = load_split(corpus, "train")
    train = build(train_cases, lens)
    model = fit("logistic", 0.1, train, list(range(len(train.cases))))
    coef = [float(v) for v in model.model.coef_[0]]
    mean = [float(v) for v in model.scaler.mean_]
    scale = [float(v) for v in model.scaler.scale_]
    intercept = float(model.model.intercept_[0])
    digest = hashlib.sha256(json.dumps([coef, mean, scale, intercept]).encode()).hexdigest()

    def rows(values: list[float]) -> str:
        return "\n".join(f"    {v!r}," for v in values)

    text = (
        '"""Frozen constants of the v0.7 candidate D (logistic regression, C=0.1, 15 features).\n\n'
        "Generated by ``python -m evaluation.run_v07 freeze`` from the v0.3 training records\n"
        "(aggregate constants only; no case data). See ``docs/releases/v0.7``.\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        f"TRAIN_CASES = {len(train.cases)}\n"
        f'WEIGHTS_SHA256 = "{digest}"\n'
        "FEATURES: tuple[str, ...] = (\n"
        + "\n".join(f'    "{n}",' for n in FEATURE_ORDER)
        + "\n)\nMEAN: tuple[float, ...] = (\n"
        + rows(mean)
        + "\n)\nSCALE: tuple[float, ...] = (\n"
        + rows(scale)
        + "\n)\nCOEFFICIENTS: tuple[float, ...] = (\n"
        + rows(coef)
        + f"\n)\nINTERCEPT = {intercept!r}\n"
    )
    target.write_text(text, encoding="utf-8")
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    runner.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    runner.add_argument("--label", default="v0.7-dev")
    freezer = sub.add_parser("freeze")
    freezer.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    freezer.add_argument("--target", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze(args.corpus_root, args.target))
        return 0
    run(args.corpus_root, args.label, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
