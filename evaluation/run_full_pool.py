#!/usr/bin/env python
"""v0.4 full-repository experiment driver (evaluator-only, aggregate-only output).

Pools every function/method of each verified failing tree, scores the methods fixed
in ``docs/releases/v0.4/selection.md``, applies the pre-registered selection and STOP
rule, and writes only an aggregate report (owner-only permissions). It trains
nothing, publishes nothing and prints no per-case data.

Run in the isolated evaluator environment (torch, sentence-transformers)::

    HF_HUB_OFFLINE=1 python -m evaluation.run_full_pool \\
      --records R.json --records-sha256 <hex> --source-manifest S.json \\
      --train-manifest T.json --tree-manifest F.json --dev \\
      --model v01=/path/model_final.safetensors:<sha256> \\
      --model v02b=/path/model_final.safetensors:<sha256>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.full_pool_methods import (
    QUERY_PREFIX,
    CaseComponents,
    build_components,
    fused_scores,
    order_by,
    result_for,
)
from evaluation.independent_runner import (
    _percentile,
    _rss_gib,
    _scores,
    paired_bootstrap,
)
from evaluation.private_adapter import PrivateCase, PrivateInputError
from evaluation.product_pool import ProductBaseline, load_product_cases
from evaluation.v3_bridge import MODEL_ID, MODEL_REVISION, parse_query

from failurelens import FailureLens

DEFAULT_OUT = Path.home() / ".local" / "share" / "failurelens" / "v04-reports"
DEFAULT_CACHE = Path.home() / ".local" / "share" / "failurelens" / "v04-embedding-cache"
W_E, W_L, W_S, W_G = (0, 0.5, 1), (0, 0.5, 1), (0, 0.25, 0.5), (0, 0.25, 0.5)
GRID = tuple(w for w in itertools.product(W_E, W_L, W_S, W_G) if any(value != 0 for value in w))
MIN_MRR_GAIN = 0.05
MAX_SEQ_LENGTH = 512
COMPARATORS = ("S0", "L0", "L1")

Metric = tuple[list[float], list[float]]  # per-case (mrr, top1)


def _reject(reason: str) -> None:
    raise PrivateInputError(f"full-pool driver rejected: {reason}") from None


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def per_case_metrics(cases: Sequence[PrivateCase], orders: Sequence[Sequence[str]]) -> Metric:
    mrr: list[float] = []
    top1: list[float] = []
    for case, ordered in zip(cases, orders, strict=True):
        scores = _scores(case, result_for(case.request, ordered))
        if scores["mrr"] is None or scores["top1"] is None:
            _reject("function labels required")
        mrr.append(float(scores["mrr"]))
        top1.append(float(scores["top1"]))
    return mrr, top1


def method_key(name: str, weights: tuple[float, ...] | None) -> str:
    return name if weights is None else f"{name}|{','.join(f'{w:g}' for w in weights)}"


def select_best(
    metrics: dict[str, Metric],
    candidates: Sequence[str],
    indices: Sequence[int] | None = None,
) -> str:
    """Highest MRR, then Top-1, then fewer non-zero weights, then smaller key."""

    def nonzero(key: str) -> int:
        if "|" not in key:
            return 0
        return sum(float(w) != 0 for w in key.split("|", 1)[1].split(","))

    def rank(key: str) -> tuple[float, float, int, str]:
        mrr, top1 = metrics[key]
        chosen = range(len(mrr)) if indices is None else indices
        return (
            -_mean([mrr[i] for i in chosen]),
            -_mean([top1[i] for i in chosen]),
            nonzero(key),
            key,
        )

    return min(candidates, key=rank)


def leave_one_repo_out(
    strata: Sequence[str], metrics: dict[str, Metric], candidates: Sequence[str]
) -> list[float]:
    """Per-case MRR when each repository is scored with a config chosen without it."""
    result = [0.0] * len(strata)
    for repo in sorted(set(strata)):
        train = [i for i, s in enumerate(strata) if s != repo]
        chosen = select_best(metrics, candidates, train)
        for i, s in enumerate(strata):
            if s == repo:
                result[i] = metrics[chosen][0][i]
    return result


def stop_rule(
    best_mrr: Sequence[float], comparator_mrr: Sequence[float], strata: Sequence[str]
) -> dict[str, Any]:
    interval = paired_bootstrap(strata, comparator_mrr, best_mrr)["delta"]
    delta = _mean(best_mrr) - _mean(comparator_mrr)
    passed = bool(interval is not None and delta >= MIN_MRR_GAIN and interval[0] > 0)
    return {
        "mrr_delta_vs_lexical_full": round(delta, 4),
        "ci95": None if interval is None else [round(v, 4) for v in interval],
        "required_gain": MIN_MRR_GAIN,
        "pass": passed,
    }


def summarize(
    name: str, metric: Metric, strata: Sequence[str], reference: Metric
) -> dict[str, Any]:
    mrr, top1 = metric
    boot_mrr = paired_bootstrap(strata, reference[0], mrr)
    boot_top1 = paired_bootstrap(strata, reference[1], top1)
    return {
        "method": name,
        "function_mrr": round(_mean(mrr), 4),
        "function_top1": round(_mean(top1), 4),
        "delta_mrr_vs_L1": round(_mean(mrr) - _mean(reference[0]), 4),
        "delta_mrr_ci95": _round(boot_mrr["delta"]),
        "delta_top1_vs_L1": round(_mean(top1) - _mean(reference[1]), 4),
        "delta_top1_ci95": _round(boot_top1["delta"]),
    }


def _round(interval: list[float] | None) -> list[float] | None:
    return None if interval is None else [round(v, 4) for v in interval]


# ---------------------------------------------------------------- embeddings


class EmbeddingCache:
    """Per-model vectors keyed by the SHA-256 of the exact function text."""

    def __init__(self, directory: Path, tag: str) -> None:
        import numpy as np

        self._np = np
        self._path = directory / f"{tag}.npz"
        self._index: dict[str, int] = {}
        self._rows: list[Any] = []
        self._dirty = False
        if self._path.is_file():
            data = np.load(self._path, allow_pickle=False)
            for position, key in enumerate(data["keys"].tolist()):
                self._index[key] = position
            self._rows = list(data["vectors"])

    def get(self, key: str) -> Any | None:
        position = self._index.get(key)
        return None if position is None else self._rows[position]

    def put(self, key: str, vector: Any) -> None:
        self._index[key] = len(self._rows)
        self._rows.append(self._np.asarray(vector, dtype="float32"))
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        keys = [k for k, _ in sorted(self._index.items(), key=lambda kv: kv[1])]
        self._np.savez(self._path, keys=self._np.array(keys), vectors=self._np.stack(self._rows))
        os.chmod(self._path, 0o600)
        self._dirty = False


Encoder = Callable[[list[str]], Any]  # texts -> normalized float vectors (n, d)


def load_encoder(
    artifact: Path | None, expected_sha256: str | None, cache_folder: str
) -> tuple[Encoder, int]:
    """Audited CodeRankEmbed, optionally with a Stage 2 state dict. Returns (encode, params)."""
    import torch
    from safetensors.torch import load_file

    from failurelens.inference.embedding import load_sentence_transformer

    if artifact is not None:
        with artifact.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected_sha256:
                _reject("model digest mismatch")
    encoder = load_sentence_transformer(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        device="cpu",
        cache_folder=cache_folder,
    )
    model = encoder._model
    if artifact is not None:
        model[0].model.load_state_dict(load_file(artifact, device="cpu"), strict=True)
    model.eval()
    model.max_seq_length = MAX_SEQ_LENGTH
    params = sum(p.numel() for p in model.parameters())

    def encode(texts: list[str]) -> Any:
        with torch.no_grad():
            return model.encode(
                texts, normalize_embeddings=True, batch_size=8, show_progress_bar=False
            )

    return encode, params


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def similarities(
    cases: Sequence[PrivateCase],
    encode: Encoder,
    cache: EmbeddingCache,
    prefix: str = "",
) -> tuple[list[list[float]], list[float], int, int]:
    """Cosine of query vs every candidate, per case. Also per-case seconds and counts."""
    import numpy as np

    sims: list[list[float]] = []
    seconds: list[float] = []
    embedded = 0
    pooled = 0
    for case in cases:
        started = time.perf_counter()
        query = prefix + parse_query(case.request.query)
        missing: dict[str, str] = {}
        for candidate in case.request.candidates:
            key = text_key(candidate.source_text)
            if cache.get(key) is None:
                missing[key] = candidate.source_text
        # Length-sorted batches keep padding (and CPU time) low.
        ordered = sorted(missing.items(), key=lambda kv: len(kv[1]))
        for start in range(0, len(ordered), 64):
            chunk = ordered[start : start + 64]
            vectors = encode([text for _, text in chunk])
            for (key, _), vector in zip(chunk, vectors, strict=True):
                cache.put(key, vector)
        q = np.asarray(encode([query])[0], dtype="float32")
        matrix = np.stack([cache.get(text_key(c.source_text)) for c in case.request.candidates])
        sims.append([float(v) for v in matrix @ q])
        embedded += len(missing)
        pooled += len(case.request.candidates)
        seconds.append(time.perf_counter() - started)
    return sims, seconds, embedded, pooled


# ---------------------------------------------------------------- experiment


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    artifact: Path | None
    sha256: str | None


def parse_model_arg(value: str) -> ModelSpec:
    tag, _, rest = value.partition("=")
    path, _, digest = rest.rpartition(":")
    if not tag or not path or len(digest) != 64:
        _reject("invalid --model (use tag=/path/model_final.safetensors:<sha256>)")
    return ModelSpec(tag, Path(path), digest)


def run_experiment(
    cases: Sequence[PrivateCase],
    baseline: ProductBaseline,
    components: Sequence[CaseComponents],
    sims: dict[str, list[list[float]]],
) -> dict[str, Any]:
    """Score all methods, select the best, and apply the pre-registered stop rule."""
    strata = [case.repository_id for case in cases]
    metrics: dict[str, Metric] = {}
    metrics["S0"] = per_case_metrics(cases, [baseline(case.request).ordered_ids for case in cases])
    metrics["L0"] = per_case_metrics(cases, [order_by(c.ids, c.capped) for c in components])
    metrics["L1"] = per_case_metrics(cases, [order_by(c.ids, c.lexical) for c in components])
    candidates: list[str] = []
    for model, per_case in sims.items():
        key = method_key(f"E-{model}", None)
        metrics[key] = per_case_metrics(
            cases, [order_by(c.ids, s) for c, s in zip(components, per_case, strict=True)]
        )
        candidates.append(key)
        for weights in GRID:
            fused = method_key(f"F-{model}", weights)
            metrics[fused] = per_case_metrics(
                cases,
                [
                    order_by(c.ids, fused_scores(c, s, weights))
                    for c, s in zip(components, per_case, strict=True)
                ],
            )
            candidates.append(fused)
    best = select_best(metrics, candidates)
    reference = metrics["L1"]
    rule = stop_rule(metrics[best][0], reference[0], strata)
    loro = leave_one_repo_out(strata, metrics, candidates)
    loro_rule = stop_rule(loro, reference[0], strata)
    singles = [
        summarize(k, metrics[k], strata, reference)
        for k in (*COMPARATORS, *[key for key in candidates if "|" not in key])
    ]
    best_per_model = []
    for model in sims:
        fused_keys = [k for k in candidates if k.startswith(f"F-{model}|")]
        top = select_best(metrics, fused_keys)
        best_per_model.append(summarize(top, metrics[top], strata, reference))
    pools = pool_recall(cases)
    return {
        "cases": len(cases),
        "repositories": len(set(strata)),
        "configurations_searched": len(candidates),
        "single_methods": singles,
        "best_fused_per_model": best_per_model,
        "selected": summarize(best, metrics[best], strata, reference),
        "stop_rule": rule,
        "leave_one_repo_out": {
            "function_mrr": round(_mean(loro), 4),
            "lexical_full_function_mrr": round(_mean(reference[0]), 4),
            **{k: v for k, v in loro_rule.items() if k != "pass"},
            "pass": loro_rule["pass"],
        },
        "pool_recall": pools,
    }


THRESHOLDS = {"top1": 0.4, "mrr": 0.6, "top1_delta": 0.4, "mrr_delta": 0.1}
BUDGETS = {"latency_p95_s": 5.0, "peak_rss_gib": 8.0}


def weights_from_method(method: str) -> tuple[float, float, float, float]:
    """Read the fixed weights of the dev-selected method (``F-<model>|a,b,c,d``)."""
    head, _, raw = method.partition("|")
    values = tuple(float(v) for v in raw.split(",")) if raw else ()
    if not head.startswith("F-") or len(values) != 4:
        _reject("final evaluation needs a fused method from the dev report")
    return values[0], values[1], values[2], values[3]


def evaluate_final(
    cases: Sequence[PrivateCase],
    baseline: ProductBaseline,
    components: Sequence[CaseComponents],
    weights: tuple[float, float, float, float],
) -> dict[str, Any]:
    """One fixed configuration vs the shipped baseline (S0) and lexical-full (L1).

    The configuration must not use the embedding term (no model is loaded here).
    """
    if weights[0] != 0:
        _reject("final evaluation without a model requires w_e = 0")
    strata = [case.repository_id for case in cases]
    metrics = {
        "S0": per_case_metrics(cases, [baseline(case.request).ordered_ids for case in cases]),
        "L1": per_case_metrics(cases, [order_by(c.ids, c.lexical) for c in components]),
        "selected": per_case_metrics(
            cases,
            [order_by(c.ids, fused_scores(c, [0.0] * len(c.ids), weights)) for c in components],
        ),
    }
    chosen = metrics["selected"]
    out: dict[str, Any] = {
        "cases": len(cases),
        "repositories": len(set(strata)),
        "weights": list(weights),
        "selected": {
            "function_mrr": round(_mean(chosen[0]), 4),
            "function_top1": round(_mean(chosen[1]), 4),
        },
        "comparisons": {},
        "pool_recall": pool_recall(cases),
    }
    for name in ("S0", "L1"):
        reference = metrics[name]
        mrr_ci = paired_bootstrap(strata, reference[0], chosen[0])
        top1_ci = paired_bootstrap(strata, reference[1], chosen[1])
        out["comparisons"][name] = {
            "reference_mrr": round(_mean(reference[0]), 4),
            "reference_top1": round(_mean(reference[1]), 4),
            "mrr_delta": round(_mean(chosen[0]) - _mean(reference[0]), 4),
            "mrr_delta_ci95": _round(mrr_ci["delta"]),
            "top1_delta": round(_mean(chosen[1]) - _mean(reference[1]), 4),
            "top1_delta_ci95": _round(top1_ci["delta"]),
        }
    top1, mrr = _mean(chosen[1]), _mean(chosen[0])
    checks: dict[str, bool] = {
        "top1_ge_0.4": top1 >= THRESHOLDS["top1"],
        "mrr_ge_0.6": mrr >= THRESHOLDS["mrr"],
    }
    for name in ("S0", "L1"):
        comparison = out["comparisons"][name]
        checks[f"top1_gain_vs_{name}_ge_0.4"] = comparison["top1_delta"] >= THRESHOLDS["top1_delta"]
        checks[f"mrr_gain_vs_{name}_ge_0.1"] = comparison["mrr_delta"] >= THRESHOLDS["mrr_delta"]
    checks["cases_ge_30"] = len(cases) >= 30
    checks["repositories_ge_3"] = len(set(strata)) >= 3
    out["threshold_checks"] = checks
    return out


def pool_recall(cases: Sequence[PrivateCase]) -> dict[str, float | int]:
    sizes = [len(case.request.candidates) for case in cases]
    function_hits = sum(
        bool(
            case.positive_function_ids
            and case.positive_function_ids & {c.function_id for c in case.request.candidates}
        )
        for case in cases
    )
    return {
        "function_recall": round(function_hits / len(cases), 4),
        "pool_size_median": int(statistics.median(sizes)),
        "pool_size_max": max(sizes),
    }


def write_report(report: dict[str, Any], out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return path


def _timing(values: Sequence[float]) -> dict[str, float]:
    return {
        "p95_s": round(_percentile(list(values), 0.95), 3),
        "max_s": round(max(values), 3),
        "median_s": round(statistics.median(values), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--tree-manifest", type=Path, required=True)
    parser.add_argument("--dev", action="store_true", help="relax the 30-case/3-repo floor")
    parser.add_argument("--model", action="append", default=[], type=parse_model_arg)
    parser.add_argument("--cache-folder", default=str(Path.home() / ".cache/huggingface/hub"))
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label", default="v0.4-dev")
    parser.add_argument(
        "--final-from",
        type=Path,
        help="dev report whose selected fused method is evaluated once, without searching",
    )
    args = parser.parse_args(argv)

    started_all = time.perf_counter()
    cases, baseline = load_product_cases(
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
    lens = FailureLens.from_baseline()
    if args.final_from is not None:
        if args.dev:
            _reject("the final evaluation must not use --dev")
        selected = json.loads(args.final_from.read_text(encoding="utf-8"))["selected"]["method"]
        weights = weights_from_method(selected)
        components = [build_components(case.request, lens.parsers.parse) for case in cases]
        rank_started = time.perf_counter()
        report = evaluate_final(cases, baseline, components, weights)
        tree_entries = json.loads(args.tree_manifest.read_text(encoding="utf-8"))["cases"]
        index_seconds = []
        for entry in tree_entries.values():
            began = time.perf_counter()
            lens.index(Path(entry["root"]))
            index_seconds.append(time.perf_counter() - began)
        rank_seconds = []
        for case in cases:
            began = time.perf_counter()
            comp = build_components(case.request, lens.parsers.parse)
            order_by(comp.ids, fused_scores(comp, [0.0] * len(comp.ids), weights))
            rank_seconds.append(time.perf_counter() - began)
        per_case = [i + r for i, r in zip(index_seconds, rank_seconds, strict=False)]
        report["label"] = args.label
        report["selected_method_from_dev"] = selected
        report["generated_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        report["timing"] = {
            "index": _timing(index_seconds),
            "rank_components": _timing(rank_seconds),
            "index_plus_rank": _timing(per_case),
        }
        report["resources"] = {
            "peak_rss_gib": round(_rss_gib(), 3),
            "model_parameters": 0,
            "total_seconds": round(time.perf_counter() - rank_started, 1),
        }
        report["threshold_checks"]["latency_p95_le_5s"] = (
            report["timing"]["index_plus_rank"]["p95_s"] <= BUDGETS["latency_p95_s"]
        )
        report["threshold_checks"]["rss_le_8gib"] = (
            report["resources"]["peak_rss_gib"] <= BUDGETS["peak_rss_gib"]
        )
        path = write_report(report, args.out_dir, f"{args.label}.json")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "selected",
                        "comparisons",
                        "threshold_checks",
                        "pool_recall",
                        "timing",
                        "resources",
                    )
                },
                indent=2,
            )
        )
        print(f"report: {path}")
        return 0
    components = [build_components(case.request, lens.parsers.parse) for case in cases]
    tree_entries = json.loads(args.tree_manifest.read_text(encoding="utf-8"))["cases"]
    index_seconds: list[float] = []
    for entry in tree_entries.values():
        began = time.perf_counter()
        lens.index(Path(entry["root"]))
        index_seconds.append(time.perf_counter() - began)

    variants: list[tuple[str, ModelSpec | None, str]] = [
        ("base", None, ""),
        ("basepfx", None, QUERY_PREFIX),
        *[(m.tag, m, "") for m in args.model],
    ]
    sims: dict[str, list[list[float]]] = {}
    embed_seconds: dict[str, list[float]] = {}
    embed_counts: dict[str, dict[str, int]] = {}
    params: dict[str, int] = {}
    loaded: dict[str, tuple[Encoder, EmbeddingCache]] = {}
    for tag, spec, prefix in variants:
        weights_tag = "base" if spec is None else spec.tag
        if weights_tag not in loaded:
            encode, count = load_encoder(
                None if spec is None else spec.artifact,
                None if spec is None else spec.sha256,
                args.cache_folder,
            )
            loaded[weights_tag] = (encode, EmbeddingCache(args.embedding_cache, weights_tag))
            params[weights_tag] = count
        encode, cache = loaded[weights_tag]
        per_case, seconds, embedded, pooled = similarities(cases, encode, cache, prefix)
        cache.save()
        sims[tag] = per_case
        embed_seconds[tag] = seconds
        embed_counts[tag] = {"embedded_functions": embedded, "pooled_functions": pooled}

    report = run_experiment(cases, baseline, components, sims)
    report["label"] = args.label
    report["generated_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    report["timing"] = {
        "index": _timing(index_seconds),
        "embedding_per_case": {tag: _timing(values) for tag, values in embed_seconds.items()},
        "embedding_counts": embed_counts,
        "note": "per-case embedding time with a shared cache; the maximum is a cold repository",
    }
    report["resources"] = {
        "peak_rss_gib": round(_rss_gib(), 3),
        "model_parameters": params,
        "total_seconds": round(time.perf_counter() - started_all, 1),
        "python": sys.version.split()[0],
    }
    path = write_report(report, args.out_dir, f"{args.label}.json")
    print(json.dumps({k: report[k] for k in ("selected", "stop_rule", "pool_recall")}, indent=2))
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
