#!/usr/bin/env python
"""v0.6 learned-reranker experiment (evaluator-only, aggregate-only output).

Trains the six candidates fixed in ``docs/releases/v0.6/selection-rule.md`` on the v0.3
TRAIN records, picks one on the v0.3 DEV records and applies the pre-registered rules
R1-R5. It never reads Gate C data, never runs a final evaluation, publishes nothing and
prints no per-case data. Reports and artifacts are owner-only local files.

Experiment environment (scikit-learn, and torch for the optional embedding group)::

    python -m evaluation.run_reranker prepare-train
    python -m evaluation.run_reranker run --v02b /path/model_final.safetensors:<sha256>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from evaluation.full_pool_methods import (
    CaseComponents,
    build_components,
    fused_scores,
    order_by,
)
from evaluation.independent_runner import _percentile, _rss_gib, paired_bootstrap
from evaluation.private_adapter import PrivateCase, PrivateInputError
from evaluation.product_pool import ProductBaseline, load_product_cases
from evaluation.reranker_features import CaseFeatures, case_features, embedding_features
from evaluation.reranker_models import (
    CONFIGS,
    FittedReranker,
    RerankerConfig,
    TrainCase,
    design_matrix,
    fit_reranker,
    save_artifact,
)
from evaluation.run_full_pool import (
    DEFAULT_CACHE,
    EmbeddingCache,
    Metric,
    ModelSpec,
    _mean,
    _round,
    load_encoder,
    parse_model_arg,
    per_case_metrics,
    pool_recall,
    similarities,
    write_report,
)

from failurelens import FailureLens

DEFAULT_CORPUS = Path.home() / ".local" / "share" / "failurelens" / "v03-corpus"
DEFAULT_OUT = Path.home() / ".local" / "share" / "failurelens" / "v06-reports"
DEFAULT_RUNS = Path.home() / ".local" / "share" / "failurelens" / "v06-runs"
V5_WEIGHTS = (0.0, 0.0, 0.25, 0.25)  # w_e, w_l, w_s, w_g of the v0.5 default
V5_DEV_MRR_IN_V05_REPORT = 0.4913
MIN_MRR_GAIN = 0.03
MAX_REPO_REGRESSION = 0.05
MAX_P95_SECONDS = 5.0
MAX_RSS_GIB = 8.0

CaseMetric = Metric  # per-case (mrr, top1)


def _reject(reason: str) -> None:
    raise PrivateInputError(f"reranker driver rejected: {reason}") from None


# ------------------------------------------------------------------ pure selection logic


def select_candidate(
    metrics: dict[str, CaseMetric],
    keys: Sequence[str],
    uses_embedding: dict[str, bool],
    indices: Sequence[int] | None = None,
) -> str:
    """Highest MRR, then Top-1, then candidates without the embedding group, then key."""

    def rank(key: str) -> tuple[float, float, bool, str]:
        mrr, top1 = metrics[key]
        chosen = range(len(mrr)) if indices is None else indices
        return (
            -_mean([mrr[i] for i in chosen]),
            -_mean([top1[i] for i in chosen]),
            uses_embedding[key],
            key,
        )

    return min(keys, key=rank)


def leave_one_repo_out_selection(
    strata: Sequence[str],
    metrics: dict[str, CaseMetric],
    keys: Sequence[str],
    uses_embedding: dict[str, bool],
) -> list[float]:
    """Per-case MRR when each dev repository is scored by a candidate chosen without it."""
    result = [0.0] * len(strata)
    for repository in sorted(set(strata)):
        others = [i for i, s in enumerate(strata) if s != repository]
        chosen = select_candidate(metrics, keys, uses_embedding, others)
        for i, s in enumerate(strata):
            if s == repository:
                result[i] = metrics[chosen][0][i]
    return result


def per_repository_mrr(strata: Sequence[str], mrr: Sequence[float]) -> dict[str, float]:
    return {
        repository: round(
            _mean([m for m, s in zip(mrr, strata, strict=True) if s == repository]), 4
        )
        for repository in sorted(set(strata))
    }


def apply_rules(
    *,
    best_mrr: Sequence[float],
    v5_mrr: Sequence[float],
    strata: Sequence[str],
    loro_mrr: Sequence[float],
    cv_mrr: float,
    v5_train_mrr: float,
    p95_seconds: float,
    peak_rss_gib: float,
) -> dict[str, Any]:
    """The five rules of ``selection-rule.md``; all must hold."""
    interval = paired_bootstrap(strata, v5_mrr, best_mrr)["delta"]
    delta = _mean(best_mrr) - _mean(v5_mrr)
    r1 = bool(interval is not None and delta >= MIN_MRR_GAIN and interval[0] > 0)
    best_by_repo = per_repository_mrr(strata, best_mrr)
    v5_by_repo = per_repository_mrr(strata, v5_mrr)
    worst = min(best_by_repo[r] - v5_by_repo[r] for r in best_by_repo)
    r2 = bool(worst >= -MAX_REPO_REGRESSION)
    loro_delta = _mean(loro_mrr) - _mean(v5_mrr)
    r3 = bool(loro_delta >= MIN_MRR_GAIN)
    r4 = bool(cv_mrr - v5_train_mrr > 0)
    r5 = bool(p95_seconds <= MAX_P95_SECONDS and peak_rss_gib <= MAX_RSS_GIB)
    return {
        "R1_gain_over_v5": {
            "mrr_delta": round(delta, 4),
            "ci95": _round(interval),
            "required": MIN_MRR_GAIN,
            "pass": r1,
        },
        "R2_no_repo_regression": {
            "worst_repo_delta": round(worst, 4),
            "allowed": -MAX_REPO_REGRESSION,
            "by_repository_candidate": best_by_repo,
            "by_repository_v5": v5_by_repo,
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
        "R5_speed_and_memory": {
            "p95_seconds": round(p95_seconds, 3),
            "peak_rss_gib": round(peak_rss_gib, 3),
            "limits": {"p95_seconds": MAX_P95_SECONDS, "peak_rss_gib": MAX_RSS_GIB},
            "pass": r5,
        },
        "all_pass": bool(r1 and r2 and r3 and r4 and r5),
    }


# ------------------------------------------------------------------------- data loading


def split_paths(root: Path, split: str) -> dict[str, Path]:
    """Records and evaluator manifests of one split (train uses the prepared train-eval set)."""
    if split == "dev":
        return {
            "records": root / "dev" / "records.json",
            "source": root / "dev" / "source_manifest.json",
            "train": root / "dev" / "train_manifest.json",
            "trees": root / "dev" / "tree_manifest.json",
        }
    if split == "train":
        return {
            "records": root / "train" / "records.json",
            "source": root / "train-eval" / "source_manifest.json",
            "train": root / "train-eval" / "train_manifest.json",
            "trees": root / "train-eval" / "tree_manifest.json",
        }
    raise ValueError(split)


def prepare_train_eval(root: Path) -> dict[str, int]:
    """Create failing-state trees and evaluator manifests for the TRAIN records.

    The trees live next to the dev trees under ``<root>/trees`` (record IDs are distinct);
    the corpus digest covers records only and is unchanged. The train-side approval file
    lists the DEV repositories so that the evaluator's train/eval disjointness check holds
    when TRAIN records are loaded as evaluation cases for feature extraction.
    """
    from training.corpus.build import GeneratedCase, Workspace, write_dev_trees

    records = json.loads((root / "train" / "records.json").read_text(encoding="utf-8"))
    mutations = json.loads((root / "train" / "mutations.json").read_text(encoding="utf-8"))
    cases = [GeneratedCase(r, mutations[r["record_id"]]) for r in records]
    out_dir = root / "train-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(root, Path(sys.executable))
    write_dev_trees(workspace, cases, out_dir)
    dev_records = json.loads((root / "dev" / "records.json").read_text(encoding="utf-8"))
    dev_repositories = sorted({r["source_repository"] for r in dev_records})
    (out_dir / "train_manifest.json").write_text(
        json.dumps(
            {
                "approval_id": "owner-delegated-v06-feature-extraction",
                "training_repositories": dev_repositories,
            }
        ),
        encoding="utf-8",
    )
    return {
        "train_cases": len(cases),
        "repositories": len({r["source_repository"] for r in records}),
    }


def load_split(root: Path, split: str) -> tuple[list[PrivateCase], ProductBaseline]:
    import hashlib

    paths = split_paths(root, split)
    digest = hashlib.sha256(paths["records"].read_bytes()).hexdigest()
    cases, baseline = load_product_cases(
        paths["records"],
        digest,
        paths["source"],
        paths["train"],
        paths["trees"],
        top_k=5,
        production=False,
        allow_ambiguous_retrieved_names=True,
        pool="full",
    )
    return list(cases), baseline


def positive_indices(case: PrivateCase) -> tuple[int, ...]:
    positives = case.positive_function_ids or frozenset()
    return tuple(i for i, c in enumerate(case.request.candidates) if c.function_id in positives)


@dataclass
class SplitData:
    cases: list[PrivateCase]
    features: list[CaseFeatures]
    components: list[CaseComponents]
    g2: list[np.ndarray | None]
    feature_seconds: list[float]

    @property
    def strata(self) -> list[str]:
        return [c.repository_id for c in self.cases]


def build_split(
    cases: list[PrivateCase], lens: FailureLens, sims: list[list[float]] | None
) -> SplitData:
    features: list[CaseFeatures] = []
    components: list[CaseComponents] = []
    g2: list[np.ndarray | None] = []
    seconds: list[float] = []
    for index, case in enumerate(cases):
        started = time.perf_counter()
        feature = case_features(case.request, lens.parsers.parse)
        seconds.append(time.perf_counter() - started)
        features.append(feature)
        components.append(build_components(case.request, lens.parsers.parse))
        g2.append(None if sims is None else embedding_features(sims[index], feature.ids))
    return SplitData(cases, features, components, g2, seconds)


def train_cases_of(data: SplitData, indices: Sequence[int]) -> list[TrainCase]:
    return [
        TrainCase(
            data.features[i].g1,
            data.g2[i],
            positive_indices(data.cases[i]),
            data.features[i].ids,
            i,
        )
        for i in indices
    ]


def order_with(model: FittedReranker, data: SplitData, i: int) -> tuple[str, ...]:
    matrix = design_matrix(model.config, data.features[i].g1, data.g2[i])
    return order_by(data.features[i].ids, [float(v) for v in model.score(matrix)])


def v5_order(components: CaseComponents) -> tuple[str, ...]:
    return order_by(
        components.ids, fused_scores(components, [0.0] * len(components.ids), V5_WEIGHTS)
    )


def metrics_for(data: SplitData, orders: Sequence[Sequence[str]]) -> CaseMetric:
    return per_case_metrics(data.cases, orders)


def cross_validate(config: RerankerConfig, train: SplitData) -> tuple[CaseMetric, float]:
    """Leave-one-repository-out over the TRAIN records; returns per-case metrics (case order)."""
    strata = train.strata
    mrr = [0.0] * len(strata)
    top1 = [0.0] * len(strata)
    started = time.perf_counter()
    for repository in sorted(set(strata)):
        fit_idx = [i for i, s in enumerate(strata) if s != repository]
        hold = [i for i, s in enumerate(strata) if s == repository]
        model = fit_reranker(config, train_cases_of(train, fit_idx))
        orders = [order_with(model, train, i) for i in hold]
        held_mrr, held_top1 = per_case_metrics([train.cases[i] for i in hold], orders)
        for position, i in enumerate(hold):
            mrr[i], top1[i] = held_mrr[position], held_top1[position]
    return (mrr, top1), time.perf_counter() - started


def summarize(
    name: str, metric: CaseMetric, strata: Sequence[str], v5: CaseMetric
) -> dict[str, Any]:
    interval = paired_bootstrap(strata, v5[0], metric[0])["delta"]
    top1_interval = paired_bootstrap(strata, v5[1], metric[1])["delta"]
    return {
        "method": name,
        "function_mrr": round(_mean(metric[0]), 4),
        "function_top1": round(_mean(metric[1]), 4),
        "delta_mrr_vs_v5": round(_mean(metric[0]) - _mean(v5[0]), 4),
        "delta_mrr_ci95": _round(interval),
        "delta_top1_vs_v5": round(_mean(metric[1]) - _mean(v5[1]), 4),
        "delta_top1_ci95": _round(top1_interval),
        "mrr_by_repository": per_repository_mrr(strata, metric[0]),
    }


def _p95(values: Sequence[float]) -> float:
    return _percentile(list(values), 0.95)


def cold_embedding_seconds(dev: SplitData, encode: Any, tag: str) -> dict[str, float]:
    """Seconds to embed one dev case of each repository with an empty cache (cold start)."""
    seen: dict[str, float] = {}
    for case in dev.cases:
        if case.repository_id in seen:
            continue
        with tempfile.TemporaryDirectory(prefix="failurelens-cold-") as scratch:
            cache = EmbeddingCache(Path(scratch), tag)
            _, seconds, _, _ = similarities([case], encode, cache)
        seen[case.repository_id] = seconds[0]
    return seen


# ----------------------------------------------------------------------------- experiment


def run_experiment(
    train: SplitData,
    dev: SplitData,
    baseline: ProductBaseline,
    index_seconds: Sequence[float],
    cold_embed: dict[str, float],
    runs_dir: Path | None,
    label: str,
    configs: Sequence[RerankerConfig] = CONFIGS,
) -> dict[str, Any]:
    strata = dev.strata
    uses_embedding = {c.key: c.uses_embedding for c in configs}
    keys = [c.key for c in configs]
    v5_dev = metrics_for(dev, [v5_order(c) for c in dev.components])
    s0_dev = metrics_for(dev, [baseline(c.request).ordered_ids for c in dev.cases])
    l1_dev = metrics_for(dev, [order_by(c.ids, c.lexical) for c in dev.components])
    v5_train = metrics_for(train, [v5_order(c) for c in train.components])

    fitted: dict[str, FittedReranker] = {}
    dev_metrics: dict[str, CaseMetric] = {}
    dev_seconds: dict[str, list[float]] = {}
    fit_seconds: dict[str, float] = {}
    cv_metrics: dict[str, CaseMetric] = {}
    cv_seconds: dict[str, float] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    all_train = list(range(len(train.cases)))
    for config in configs:
        started = time.perf_counter()
        model = fit_reranker(config, train_cases_of(train, all_train))
        fit_seconds[config.key] = time.perf_counter() - started
        fitted[config.key] = model
        orders: list[tuple[str, ...]] = []
        per_case: list[float] = []
        for i in range(len(dev.cases)):
            began = time.perf_counter()
            orders.append(order_with(model, dev, i))
            per_case.append(time.perf_counter() - began)
        dev_metrics[config.key] = metrics_for(dev, orders)
        dev_seconds[config.key] = per_case
        cv_metrics[config.key], cv_seconds[config.key] = cross_validate(config, train)
        if runs_dir is not None:
            path = runs_dir / label / f"candidate-{config.key.lower()}" / "model.pkl"
            artifacts[config.key] = {"sha256": save_artifact(model, path)}

    best = select_candidate(dev_metrics, keys, uses_embedding)
    best_config = next(c for c in configs if c.key == best)
    loro = leave_one_repo_out_selection(strata, dev_metrics, keys, uses_embedding)

    def case_seconds(config: RerankerConfig) -> list[float]:
        total: list[float] = []
        for i, case in enumerate(dev.cases):
            value = index_seconds[i] + dev.feature_seconds[i] + dev_seconds[config.key][i]
            if config.uses_embedding:
                value += cold_embed.get(case.repository_id, 0.0)
            total.append(value)
        return total

    best_p95 = _p95(case_seconds(best_config))
    rss = _rss_gib()
    cv_best = _mean(cv_metrics[best][0])
    rules = apply_rules(
        best_mrr=dev_metrics[best][0],
        v5_mrr=v5_dev[0],
        strata=strata,
        loro_mrr=loro,
        cv_mrr=cv_best,
        v5_train_mrr=_mean(v5_train[0]),
        p95_seconds=best_p95,
        peak_rss_gib=rss,
    )
    candidates = []
    for config in configs:
        row = summarize(f"candidate-{config.key}", dev_metrics[config.key], strata, v5_dev)
        row.update(config.describe())
        row.update(
            {
                "fit_seconds": round(fit_seconds[config.key], 2),
                "train_cv_function_mrr": round(_mean(cv_metrics[config.key][0]), 4),
                "train_cv_function_top1": round(_mean(cv_metrics[config.key][1]), 4),
                "train_cv_delta_vs_v5": round(
                    _mean(cv_metrics[config.key][0]) - _mean(v5_train[0]), 4
                ),
                "dev_p95_seconds": round(_p95(case_seconds(config)), 3),
                "coefficients_standardized": fitted[config.key].coefficients(),
                "artifact": artifacts.get(config.key),
            }
        )
        candidates.append(row)
    best_g1 = select_candidate(
        dev_metrics, [k for k in keys if not uses_embedding[k]], uses_embedding
    )
    best_g2_keys = [k for k in keys if uses_embedding[k]]
    embedding_ablation = {
        "best_without_embedding": best_g1,
        "best_without_embedding_mrr": round(_mean(dev_metrics[best_g1][0]), 4),
        "best_with_embedding": select_candidate(dev_metrics, best_g2_keys, uses_embedding)
        if best_g2_keys
        else None,
        "best_with_embedding_mrr": round(
            _mean(dev_metrics[select_candidate(dev_metrics, best_g2_keys, uses_embedding)][0]), 4
        )
        if best_g2_keys
        else None,
    }
    return {
        "label": label,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_cases": len(train.cases),
        "train_repositories": len(set(train.strata)),
        "dev_cases": len(dev.cases),
        "dev_repositories": len(set(strata)),
        "references": {
            "V5": summarize("V5", v5_dev, strata, v5_dev),
            "S0": summarize("S0", s0_dev, strata, v5_dev),
            "L1": summarize("L1", l1_dev, strata, v5_dev),
            "v5_reproduces_v05_report": round(_mean(v5_dev[0]), 4) == V5_DEV_MRR_IN_V05_REPORT,
        },
        "candidates": candidates,
        "selected": best,
        "selected_summary": summarize(f"candidate-{best}", dev_metrics[best], strata, v5_dev),
        "rules": rules,
        "verdict": "pass" if rules["all_pass"] else "stop",
        "leave_one_repo_out_selection": {
            "function_mrr": round(_mean(loro), 4),
            "mrr_delta_vs_v5": round(_mean(loro) - _mean(v5_dev[0]), 4),
        },
        "embedding_ablation": embedding_ablation,
        "pool_recall_dev": pool_recall(dev.cases),
        "pool_recall_train": pool_recall(train.cases),
        "timing": {
            "index_p95_s": round(_p95(index_seconds), 3),
            "features_p95_s": round(_p95(dev.feature_seconds), 3),
            "cold_embedding_s_by_repository": {k: round(v, 2) for k, v in cold_embed.items()},
        },
        "resources": {"peak_rss_gib": round(rss, 3), "python": sys.version.split()[0]},
    }


# ------------------------------------------------------------------------------------ CLI


def _build_sims(
    spec: ModelSpec,
    cache_dir: Path,
    cache_folder: str,
    splits: Sequence[list[PrivateCase]],
) -> tuple[list[list[list[float]]], Any]:
    encode, _ = load_encoder(spec.artifact, spec.sha256, cache_folder)
    cache = EmbeddingCache(cache_dir, spec.tag)
    results: list[list[list[float]]] = []
    for cases in splits:
        sims, _, _, _ = similarities(cases, encode, cache)
        cache.save()
        results.append(sims)
    return results, encode


def _cmd_prepare(args: argparse.Namespace) -> int:
    summary = prepare_train_eval(args.corpus_root)
    print(json.dumps(summary))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    lens = FailureLens.from_baseline()
    train_cases, _ = load_split(args.corpus_root, "train")
    dev_cases, baseline = load_split(args.corpus_root, "dev")
    spec: ModelSpec | None = args.v02b
    encode: Any = None
    train_sims: list[list[float]] | None = None
    dev_sims: list[list[float]] | None = None
    if spec is not None:
        (train_sims, dev_sims), encode = _build_sims(
            spec, args.embedding_cache, args.cache_folder, [train_cases, dev_cases]
        )
    train = build_split(train_cases, lens, train_sims)
    dev = build_split(dev_cases, lens, dev_sims)
    tree_entries = json.loads(
        split_paths(args.corpus_root, "dev")["trees"].read_text(encoding="utf-8")
    )["cases"]
    index_seconds: list[float] = []
    for entry in tree_entries.values():
        began = time.perf_counter()
        lens.index(Path(entry["root"]))
        index_seconds.append(time.perf_counter() - began)
    if len(index_seconds) != len(dev.cases):
        _reject("tree manifest and dev cases differ in length")
    cold = cold_embedding_seconds(dev, encode, spec.tag) if spec is not None else {}
    configs = CONFIGS if spec is not None else tuple(c for c in CONFIGS if not c.uses_embedding)
    report = run_experiment(
        train, dev, baseline, index_seconds, cold, args.runs_dir, args.label, configs
    )
    report["resources"]["total_seconds"] = round(time.perf_counter() - started, 1)
    report["embedding_group_run"] = spec is not None
    path = write_report(report, args.out_dir, f"{args.label}.json")
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "selected_summary": report["selected_summary"],
                "rules": report["rules"],
                "verdict": report["verdict"],
            },
            indent=2,
        )
    )
    print(f"report: {path}")
    return 0


def _spec(value: str) -> ModelSpec:
    return parse_model_arg(f"v02b={value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-train", help="build failing trees for TRAIN records")
    prepare.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    prepare.set_defaults(handler=_cmd_prepare)
    run = sub.add_parser("run", help="train the six candidates, select, apply the rules")
    run.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    run.add_argument("--v02b", type=_spec, help="artifact_path:sha256 of v0.2 candidate b")
    run.add_argument("--cache-folder", default=str(Path.home() / ".cache/huggingface/hub"))
    run.add_argument("--embedding-cache", type=Path, default=DEFAULT_CACHE)
    run.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    run.add_argument("--label", default="v0.6-dev")
    run.set_defaults(handler=_cmd_run)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
