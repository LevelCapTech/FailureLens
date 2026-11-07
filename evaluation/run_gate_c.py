#!/usr/bin/env python
"""Reusable, aggregate-only driver: shipped baseline vs one model artifact.

Wires the evaluator-owned loaders (``product_pool``) and the three-seed
``independent_runner`` for one candidate artifact. Inputs (records, manifests,
failing trees, weights) stay local and are passed as arguments; only the
aggregate report is written, with owner-only permissions, to a private output
directory. The console summary contains numbers only.

This is a measurement tool. It does not decide Gate C, publish anything, or
download data. The model base-cache must already hold the audited revision.

Usage (isolated evaluator environment with the ``ml`` and ``train`` extras)::

    HF_HUB_OFFLINE=1 python -m evaluation.run_gate_c \\
      --records R.json --records-sha256 <hex> --source-manifest S.json \\
      --train-manifest T.json --tree-manifest F.json \\
      --model-artifact model_final.safetensors --model-sha256 <hex> \\
      --model-label v0.2-candidate-1 --cache-folder ~/.cache/huggingface/hub

``--dev`` relaxes only the 30-case / 3-repository production floor so a smaller
development set can be scored; such reports are labelled ``dev`` and are not
Gate C evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.independent_runner import MethodSpec, run_independent
from evaluation.private_adapter import PrivateInputError, load_private_cases
from evaluation.product_pool import load_product_cases
from evaluation.v3_bridge import (
    MODEL_ID,
    MODEL_REVISION,
    V3Ranker,
    _cache_has_unsafe_weights,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path.home() / ".local" / "share" / "failurelens" / "gate-c-reports"
COMMAND_ID = "run_independent-api"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,59}")
_BLOCKED = ("private", "case", "label", "log", "source", "path", "record")

# Optional hook for experiment tracking (set by the tracking integration).
# Receives only the sanitized aggregate report and the version label.
RecordHook = Callable[[dict[str, Any], str], None]


def _reject(reason: str) -> None:
    raise PrivateInputError(f"gate-c driver rejected: {reason}") from None


def record_evaluation(report: dict[str, Any], version_label: str, hook: RecordHook | None) -> bool:
    """Forward the aggregate report to an optional tracking hook; no-op without one."""
    if hook is None:
        return False
    hook(report, version_label)
    return True


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git_head() -> tuple[str, bool]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                check=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        _reject("unavailable code revision")
    return sha, dirty


def hardware_token(device: str) -> str:
    """A short allowlist-safe description of the compute device."""
    name = "unknown-cpu"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        name = platform.processor() or name
    token = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{device}-{name}").strip("_")[:80]
    if any(word in token.lower() for word in _BLOCKED) or not token:
        token = device
    return token


def environment_versions() -> dict[str, str]:
    """Installed versions of the packages that determine the measurement."""
    from importlib import metadata

    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "sentence-transformers", "safetensors", "einops"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "absent"
    return versions


def verify_candidate(
    artifact: Path, expected_sha256: str, train_sha256: str | None
) -> tuple[MethodSpec, str]:
    """Check the weights digest and recorded provenance; return public metadata."""
    if not _HEX64.fullmatch(expected_sha256):
        _reject("invalid model digest")
    if artifact.name != "model_final.safetensors" or not artifact.is_file():
        _reject("unavailable model artifact")
    if sha256_file(artifact) != expected_sha256:
        _reject("model digest mismatch")
    try:
        meta = json.loads((artifact.parent / "experiment_metadata.json").read_text("utf-8"))
        run = meta["run_metadata"]
    except (OSError, ValueError, KeyError, TypeError):
        _reject("unavailable model provenance")
    if (
        not isinstance(run, dict)
        or run.get("model_id") != MODEL_ID
        or run.get("model_revision") != MODEL_REVISION
        or meta.get("artifact_sha256") != expected_sha256
        or (train_sha256 is not None and run.get("dataset_sha256") != train_sha256)
    ):
        _reject("model provenance mismatch")
    count = run.get("total_params")
    if type(count) is not int or count < 1 or count > 1_000_000_000:
        _reject("invalid model parameter metadata")
    dataset = run.get("dataset_sha256")
    if not isinstance(dataset, str) or not _HEX64.fullmatch(dataset):
        _reject("invalid training digest")
    name = "candidate-coderankembed"
    return MethodSpec(name, expected_sha256, count, artifact.stat().st_size), dataset


def make_candidate_factory(
    artifact: Path, expected_sha256: str, cache_folder: str, device: str
) -> Callable[[int], V3Ranker]:
    """``model_factory(seed)`` for any Stage 2 artifact on the audited base model."""
    if device not in {"cpu", "cuda"}:
        _reject("invalid device")
    snapshot = Path(cache_folder) / "models--nomic-ai--CodeRankEmbed" / "snapshots" / MODEL_REVISION
    if not snapshot.is_dir() or _cache_has_unsafe_weights(cache_folder):
        _reject("unavailable audited model cache")

    def factory(seed: int) -> V3Ranker:
        if type(seed) is not int:
            _reject("invalid seed")
        if sha256_file(artifact) != expected_sha256:
            _reject("model digest mismatch")
        try:
            import torch
            from safetensors.torch import load_file

            from failurelens.inference.embedding import load_sentence_transformer

            torch.manual_seed(seed)
            if device == "cuda":
                if not torch.cuda.is_available():
                    _reject("requested device unavailable")
                torch.cuda.manual_seed_all(seed)
            encoder = load_sentence_transformer(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=True,
                device=device,
                cache_folder=cache_folder,
            )
            model = encoder._model
            model[0].model.load_state_dict(load_file(artifact, device="cpu"), strict=True)
            model.eval()
            return V3Ranker(model)
        except PrivateInputError:
            raise
        except Exception:
            _reject("model unavailable or incompatible")

    return factory


def _label(value: str) -> str:
    if not _LABEL.fullmatch(value) or any(word in value.lower() for word in _BLOCKED):
        _reject("invalid version label")
    return value


def run_measurement(args: argparse.Namespace, hook: RecordHook | None = None) -> dict[str, Any]:
    """Load verified inputs, measure baseline vs candidate, return the aggregate report."""
    label = _label(args.model_label)
    code_sha, dirty = _git_head()
    if dirty and not args.allow_dirty:
        _reject("uncommitted code changes; commit or pass --allow-dirty")
    spec, train_digest = verify_candidate(args.model_artifact, args.model_sha256, args.train_sha256)
    baseline_kwargs: dict[str, Any] = {}
    if args.pool == "product":
        cases, baseline = load_product_cases(
            args.records,
            args.records_sha256,
            args.source_manifest,
            args.train_manifest,
            args.tree_manifest,
            top_k=args.top_k,
            production=not args.dev,
        )
        baseline_kwargs = {
            "baseline": baseline,
            "baseline_spec": MethodSpec("shipped-baseline-stacktrace-lexical", "product-v1", 0, 0),
        }
    else:
        # Labeled candidate pools; comparator is the supplied-pool lexical adapter,
        # which is NOT the shipped analyzer. Not eligible for a shipped-baseline delta.
        cases = load_private_cases(
            args.records,
            args.records_sha256,
            args.source_manifest,
            args.train_manifest,
            top_k=args.top_k,
            production=not args.dev,
        )
    factory = make_candidate_factory(
        args.model_artifact, args.model_sha256, args.cache_folder, args.device
    )
    metadata = {
        "code_sha": code_sha,
        "data_sha256": args.records_sha256,
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "model_sha256": args.model_sha256,
        "python_version": platform.python_version(),
        "hardware": hardware_token(args.device),
        "command": COMMAND_ID,
    }
    report = run_independent(
        cases,
        factory,
        model_spec=spec,
        metadata=metadata,
        **baseline_kwargs,
    )
    report["version_label"] = label
    report["pool"] = args.pool
    report["mode"] = "dev" if args.dev else "gate-c-measurement"
    report["environment"] = environment_versions()
    report["code_dirty"] = dirty
    report["training_data_sha256"] = train_digest
    report["generated_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record_evaluation(report, label, hook)
    return report


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    """Write the aggregate report with owner-only permissions (private directory)."""
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = (
        f"{report['mode']}-{report['pool']}-{report['version_label']}-{report['generated_at']}.json"
    )
    target = out_dir / name.replace(":", "")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def summarize(report: dict[str, Any]) -> str:
    """Plain, numbers-only console summary; never includes case data."""
    counts, summary = report["counts"], report["summary"]
    intervals, comparison = report["intervals"], report["threshold_comparison"]
    lines = [
        f"version: {report['version_label']}  mode: {report['mode']}  pool: {report['pool']}",
        f"cases: {counts['cases']}  repositories: {counts['repositories']}  "
        f"function-labeled: {counts['function_labeled']}",
        f"pool: abstentions={counts['retrieval_abstentions']}  "
        f"positive-function-absent={counts['positive_function_absent_from_pool']}  "
        f"function-recall={_fmt(counts['retrieved_pool_function_recall'])}",
    ]
    for method in ("baseline", "model"):
        row = summary[method]
        lines.append(
            f"{method}: top1={_fmt(row['top1']['median'])} mrr={_fmt(row['mrr']['median'])} "
            f"file_top1={_fmt(row['file_top1']['median'])} "
            f"latency_p95_s_max={_fmt(row['latency_p95_s']['max'])}"
        )
    for metric in ("top1", "mrr"):
        interval = intervals[metric]
        lines.append(
            f"{metric} 95% CI: baseline={interval['baseline']} model={interval['model']} "
            f"delta={interval['delta']}"
        )
    lines.append(f"thresholds (mechanical): {comparison}")
    lines.append(f"resources: {report['resource_comparison']}")
    lines.append("gate_c_decision: none (measurement only)")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument(
        "--tree-manifest",
        type=Path,
        required=True,
        help="failing-tree manifest (read only for --pool product)",
    )
    parser.add_argument(
        "--pool",
        choices=("product", "supplied"),
        default="product",
        help="product: shipped-baseline retrieval; supplied: labeled candidate pools",
    )
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--train-sha256", help="expected training dataset digest")
    parser.add_argument("--model-label", required=True, help="e.g. v0.1-reference")
    parser.add_argument("--cache-folder", default=str(Path.home() / ".cache/huggingface/hub"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dev", action="store_true", help="relax the 30-case/3-repo floor")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, hook: RecordHook | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_measurement(args, hook)
        target = write_report(report, args.out_dir)
    except PrivateInputError as error:
        print(str(error), file=sys.stderr)
        return 4
    print(summarize(report))
    print(f"aggregate report written (owner-only): {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
