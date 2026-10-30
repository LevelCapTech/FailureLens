"""Local-only independent measurement over already validated, identical candidate pools.

Only aggregate reports leave this module. Caller must use load_private_cases first;
ranker factories must be pinned and must never receive labels.
"""

from __future__ import annotations

import json
import math
import platform
import random
import resource
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from evaluation.private_adapter import (
    PrivateCase,
    PrivateInputError,
    Ranker,
    RankRequest,
    RankResult,
    run_ranker,
)

from failurelens.ranking.baseline import extract_error_tokens

SEEDS = (20261030, 20261031, 20261032)
BOOTSTRAP_SEED = 20261030
BOOTSTRAP_REPLICATES = 10_000
THRESHOLDS = {"top1": 0.4, "mrr": 0.6, "top1_delta": 0.4, "mrr_delta": 0.1}
BUDGETS = {"latency_p95_s": 5.0, "peak_rss_gib": 8.0, "parameters": 1_000_000_000}


def lexical_stack_baseline(request: RankRequest) -> RankResult:
    """Rank only supplied candidates by stack/path evidence plus lexical overlap.

    This is a candidate-pool adapter, not the repository-retrieving product baseline.
    Ties break by candidate ID ascending, independently of input order.
    """
    framed = json.loads(request.query)
    log = str(framed["error_log"])
    trace = str(framed.get("stack_trace") or "")
    tokens = extract_error_tokens(log)

    def score(candidate: Any) -> tuple[float, str]:
        # Source lexical overlap and stack mentions are transparent heuristics.
        lexical = min(sum(token in candidate.source_text for token in tokens), 3) / 3
        stack = int(candidate.file_id in trace) + int(
            candidate.function_id is not None and candidate.function_id.split("::")[-1] in trace
        )
        return (-(0.6 * stack + 0.4 * lexical), candidate.id)

    ordered = tuple(candidate.id for candidate in sorted(request.candidates, key=score))
    return RankResult(ordered, request.digest, request.top_k)


def _rr(ordered: Sequence[str], positives: frozenset[str]) -> float:
    for rank, item in enumerate(ordered, 1):
        if item in positives:
            return 1 / rank
    return 0.0


def _scores(case: PrivateCase, result: RankResult) -> dict[str, float | None]:
    by_id = {candidate.id: candidate for candidate in case.request.candidates}
    files = tuple(dict.fromkeys(by_id[item].file_id for item in result.ordered_ids))
    functions = tuple(
        dict.fromkeys(
            by_id[item].function_id
            for item in result.ordered_ids
            if by_id[item].function_id is not None
        )
    )
    k = case.request.top_k
    positive_files = case.positive_file_ids
    values: dict[str, float | None] = {
        "file_top1": float(bool(files and files[0] in positive_files)),
        "file_topk": float(bool(set(files[:k]) & positive_files)),
        "file_recall_at_k": len(set(files[:k]) & positive_files) / len(positive_files),
        "top1": None,
        "mrr": None,
    }
    if case.positive_function_ids is not None:
        positives = case.positive_function_ids
        values["top1"] = float(bool(functions and functions[0] in positives))
        values["mrr"] = _rr(functions, positives)
    return values


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lower = math.floor(pos)
    upper = math.ceil(pos)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def paired_bootstrap(
    strata: Sequence[str], baseline: Sequence[float | None], model: Sequence[float | None]
) -> dict[str, list[float] | None]:
    """Stratified paired case bootstrap; one observation/case, never per rerun."""
    if not (len(strata) == len(baseline) == len(model)):
        raise PrivateInputError("private evaluation rejected: missing metric samples")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        if baseline[index] is not None and model[index] is not None:
            groups[stratum].append(index)
        elif baseline[index] != model[index]:
            raise PrivateInputError("private evaluation rejected: label parity violation")
    if not groups:
        return {"baseline": None, "model": None, "delta": None}
    rng = random.Random(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {"baseline": [], "model": [], "delta": []}
    for _ in range(BOOTSTRAP_REPLICATES):
        drawn = [rng.choice(indices) for indices in groups.values() for _ in indices]
        b = statistics.mean(float(baseline[i]) for i in drawn)
        m = statistics.mean(float(model[i]) for i in drawn)
        samples["baseline"].append(b)
        samples["model"].append(m)
        samples["delta"].append(m - b)
    return {
        key: [_percentile(vals, 0.025), _percentile(vals, 0.975)] for key, vals in samples.items()
    }


@dataclass(frozen=True)
class MethodSpec:
    """Frozen public metadata; no paths, case identifiers, or source text."""

    name: str
    revision: str
    parameter_count: int | None = None
    artifact_bytes: int | None = None


def _rss_gib() -> float:
    # Linux ru_maxrss is KiB; macOS is bytes. Includes model load in this process.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if platform.system() == "Linux" else 1024**3)


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Allowlist aggregate provenance only; never copy arbitrary private strings."""
    allowed = {
        "code_sha",
        "data_sha256",
        "source_manifest_sha256",
        "train_manifest_sha256",
        "model_sha256",
        "python_version",
        "hardware",
        "command",
        "top_k",
    }
    if set(metadata) - allowed or any(not isinstance(v, (str, int)) for v in metadata.values()):
        raise PrivateInputError("private evaluation rejected: unsafe report metadata")
    import re

    digest_keys = {
        "code_sha",
        "data_sha256",
        "source_manifest_sha256",
        "train_manifest_sha256",
        "model_sha256",
    }
    blocked = ("private", "case", "label", "log", "source", "path", "record")
    for key, value in metadata.items():
        if key not in digest_keys and any(word in str(value).lower() for word in blocked):
            raise PrivateInputError("private evaluation rejected: unsafe report metadata")
        if key in digest_keys and not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(value)):
            raise PrivateInputError("private evaluation rejected: unsafe report metadata")
        if key == "top_k" and (type(value) is not int or value < 1):
            raise PrivateInputError("private evaluation rejected: unsafe report metadata")
        if key == "command" and value != "run_independent-api":
            raise PrivateInputError("private evaluation rejected: unsafe report metadata")
        if (
            key not in digest_keys
            and key not in {"top_k", "command"}
            and not re.fullmatch(r"[A-Za-z0-9_.:+-]{1,80}", str(value))
        ):
            raise PrivateInputError("private evaluation rejected: unsafe report metadata")
    return dict(metadata)


def _safe_spec(spec: MethodSpec) -> dict[str, Any]:
    import re

    blocked = ("private", "case", "label", "log", "source", "path", "record")
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", spec.name)
        or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", spec.revision)
        or any(word in spec.name.lower() or word in spec.revision.lower() for word in blocked)
    ):
        raise PrivateInputError("private evaluation rejected: unsafe method metadata")
    if spec.parameter_count is not None and spec.parameter_count < 0:
        raise PrivateInputError("private evaluation rejected: invalid parameter count")
    if spec.artifact_bytes is not None and spec.artifact_bytes < 0:
        raise PrivateInputError("private evaluation rejected: invalid artifact size")
    return {
        "name": spec.name,
        "revision": spec.revision,
        "parameter_count": spec.parameter_count,
        "artifact_bytes": spec.artifact_bytes,
    }


def run_independent(
    cases: Sequence[PrivateCase],
    model_factory: Callable[[int], Ranker] | None,
    *,
    baseline: Ranker = lexical_stack_baseline,
    baseline_spec: MethodSpec | None = None,
    model_spec: MethodSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure three fixed seeds; return strictly aggregate, committable report.

    All cases must already have passed the Issue #21 private input boundary. No
    actual model is bundled. Model factory is called once per seed before timing
    case inference; its load remains included in process peak RSS.
    """
    if not cases or any(not case.request.candidates for case in cases):
        raise PrivateInputError("private evaluation rejected: missing cases")
    safe_metadata = _safe_metadata(metadata or {})
    required = {
        "code_sha",
        "data_sha256",
        "source_manifest_sha256",
        "train_manifest_sha256",
        "model_sha256",
        "python_version",
        "hardware",
        "command",
    }
    if model_factory is not None and not required.issubset(safe_metadata):
        raise PrivateInputError("private evaluation rejected: missing provenance metadata")
    base_spec = _safe_spec(baseline_spec or MethodSpec("supplied-pool-stack-lexical", "v1", 0, 0))
    if model_factory is not None and (
        model_spec is None
        or model_spec.parameter_count is None
        or model_spec.artifact_bytes is None
    ):
        raise PrivateInputError("private evaluation rejected: missing model metadata")
    model_info = _safe_spec(model_spec) if model_spec is not None else None
    if model_factory is None and model_spec is not None:
        raise PrivateInputError("private evaluation rejected: model absent")
    strata = [case.repository_id for case in cases]
    if any(not stratum for stratum in strata):
        raise PrivateInputError("private evaluation rejected: missing repository stratum")
    if len({case.request.top_k for case in cases}) != 1:
        raise PrivateInputError("private evaluation rejected: top_k mismatch")
    runs: list[dict[str, Any]] = []
    first_scores: dict[str, list[dict[str, float | None]]] = {}
    for seed in SEEDS:
        load_start = time.perf_counter()
        try:
            model = model_factory(seed) if model_factory is not None else None
        except Exception:
            raise PrivateInputError("private evaluation rejected: ranker failed") from None
        if model_factory is not None and model is None:
            raise PrivateInputError("private evaluation rejected: ranker failed")
        model_load_s = time.perf_counter() - load_start if model is not None else None
        scores: dict[str, list[dict[str, float | None]]] = {"baseline": [], "model": []}
        latencies: dict[str, list[float]] = {"baseline": [], "model": []}
        for case in cases:
            for method, ranker in (("baseline", baseline), ("model", model)):
                if ranker is None:
                    continue
                start = time.perf_counter()
                result = run_ranker(case, ranker)
                elapsed = time.perf_counter() - start
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise PrivateInputError("private evaluation rejected: missing latency sample")
                latencies[method].append(elapsed)
                scores[method].append(_scores(case, result))
        aggregate: dict[str, Any] = {}
        for method in ("baseline", "model"):
            aggregate[method] = (
                {
                    **{
                        key: _mean([row[key] for row in scores[method]])
                        for key in ("top1", "mrr", "file_top1", "file_topk", "file_recall_at_k")
                    },
                    "latency_p95_s": _percentile(latencies[method], 0.95),
                }
                if scores[method]
                else None
            )
        rss = _rss_gib()
        if (
            not math.isfinite(rss)
            or rss <= 0
            or (model is not None and (model_load_s is None or not math.isfinite(model_load_s)))
        ):
            raise PrivateInputError("private evaluation rejected: missing resource sample")
        failure_categories = {
            method: (
                {
                    "function_top1_miss": sum(row["top1"] == 0.0 for row in scores[method]),
                    "file_topk_miss": sum(row["file_topk"] == 0.0 for row in scores[method]),
                }
                if scores[method]
                else None
            )
            for method in ("baseline", "model")
        }
        runs.append(
            {
                "seed": seed,
                "metrics": aggregate,
                "failure_categories": failure_categories,
                "peak_rss_gib": rss,
                "model_load_s": model_load_s,
            }
        )
        if not first_scores:
            first_scores = scores
    intervals = (
        {
            metric: paired_bootstrap(
                strata,
                [row[metric] for row in first_scores["baseline"]],
                [row[metric] for row in first_scores["model"]],
            )
            for metric in ("top1", "mrr")
        }
        if model_factory is not None
        else {"top1": None, "mrr": None}
    )
    summary: dict[str, Any] = {}
    for method in ("baseline", "model"):
        summary[method] = {}
        for metric in (
            "top1",
            "mrr",
            "file_top1",
            "file_topk",
            "file_recall_at_k",
            "latency_p95_s",
        ):
            values = [
                run["metrics"][method][metric] for run in runs if run["metrics"][method] is not None
            ]
            present = [float(value) for value in values if value is not None]
            summary[method][metric] = (
                {"min": min(present), "median": statistics.median(present), "max": max(present)}
                if present
                else None
            )
    deltas = {
        metric: (summary["model"][metric]["median"] - summary["baseline"][metric]["median"])
        if summary["model"][metric] is not None and summary["baseline"][metric] is not None
        else None
        for metric in ("top1", "mrr")
    }
    comparisons = {
        "top1": None
        if summary["model"]["top1"] is None
        else summary["model"]["top1"]["median"] >= THRESHOLDS["top1"],
        "mrr": None
        if summary["model"]["mrr"] is None
        else summary["model"]["mrr"]["median"] >= THRESHOLDS["mrr"],
        "top1_delta": None
        if deltas["top1"] is None
        else deltas["top1"] >= THRESHOLDS["top1_delta"],
        "mrr_delta": None if deltas["mrr"] is None else deltas["mrr"] >= THRESHOLDS["mrr_delta"],
    }
    return {
        "schema": "failurelens-independent-aggregate-v1",
        "status": "synthetic-or-unreviewed" if model_factory is not None else "model-not-run",
        "metadata": {**safe_metadata, "top_k": cases[0].request.top_k},
        "methods": {"baseline": base_spec, "model": model_info},
        "counts": {
            "cases": len(cases),
            "repositories": len(set(strata)),
            "function_labeled": sum(case.positive_function_ids is not None for case in cases),
            "function_missing": sum(case.positive_function_ids is None for case in cases),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap": "repository-stratified paired case percentile two-sided 95%",
            "latency": "per-method per-case wall time; warm inference; model factory load separate",
            "memory": "process peak RSS including model load; Linux KiB macOS bytes",
        },
        "runs": runs,
        "summary": summary,
        "intervals": intervals,
        "deltas": deltas,
        "thresholds": THRESHOLDS,
        "budgets": BUDGETS,
        "threshold_comparison": comparisons,
        "resource_comparison": {
            "latency_p95": None
            if summary["model"]["latency_p95_s"] is None
            else summary["model"]["latency_p95_s"]["max"] <= BUDGETS["latency_p95_s"],
            "peak_rss": None
            if model_factory is None
            else max(run["peak_rss_gib"] for run in runs) <= BUDGETS["peak_rss_gib"],
            "parameters": None
            if model_info is None
            else model_info["parameter_count"] <= BUDGETS["parameters"],
        },
        "gate_c_decision": None,
    }
