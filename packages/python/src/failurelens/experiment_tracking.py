"""Opt-in MLflow tracking for development experiments, separate from diagnosis."""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import os
import platform
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from failurelens.errors import MissingDependencyError

MODEL_VERSION_PATTERN = r"^v\d+\.\d+\.\d+(-[A-Za-z0-9.]+)*$"
"""Model release labels such as ``v0.1.0`` or ``v0.2.0-candidate-1``."""


class MLflowConfig(BaseModel):
    """Supplying this config explicitly enables experiment tracking."""

    model_config = ConfigDict(extra="forbid")

    tracking_uri: str | None = None
    experiment_name: str | None = Field(default=None, min_length=1)
    run_name: str | None = Field(default=None, min_length=1)
    model_version: str | None = Field(default=None, pattern=MODEL_VERSION_PATTERN)


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same opt-in flags to evaluation and training entry points."""
    group = parser.add_argument_group("optional MLflow experiment tracking")
    group.add_argument("--mlflow", action="store_true", help="Record this experiment in MLflow.")
    group.add_argument(
        "--mlflow-tracking-uri", help="Override MLFLOW_TRACKING_URI (default: local SQLite)."
    )
    group.add_argument("--mlflow-experiment", help="Override MLFLOW_EXPERIMENT_NAME.")
    group.add_argument("--mlflow-run-name", help="Optional display name for this run.")
    group.add_argument(
        "--model-version",
        help="Model version label (for example v0.2.0-candidate-1) used to compare versions.",
    )


def tracking_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> MLflowConfig | None:
    """Require explicit opt-in even when MLflow environment variables are set."""
    if not args.mlflow:
        if (
            args.mlflow_tracking_uri
            or args.mlflow_experiment
            or args.mlflow_run_name
            or args.model_version
        ):
            parser.error("MLflow options require --mlflow")
        return None
    return MLflowConfig(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
        model_version=args.model_version,
    )


def _git(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip()


def _environment() -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for package in (
        "failurelens",
        "mlflow",
        "pydantic",
        "typer",
        "torch",
        "transformers",
        "sentence-transformers",
    ):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "dependencies": dependencies,
    }


def training_diagnostics(
    train_loss: Sequence[float], validation_loss: Sequence[float]
) -> dict[str, float]:
    """Summarise train/validation curves; epochs are 1-based.

    ``diag.overfit_signal`` is 1.0 only when validation loss is worse than its
    best epoch while training loss kept improving after that epoch. It is a
    heuristic warning for the loss curves, not proof of overfitting: with few
    epochs and few examples the curves are noisy. ``diag.gap_trend`` is the
    least-squares slope of (validation - train) per epoch; positive means the
    gap is widening.
    """
    if len(train_loss) != len(validation_loss) or not train_loss:
        raise ValueError("training and validation histories must be non-empty and equal length")
    values = [*train_loss, *validation_loss]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("training diagnostics require finite losses")
    count = len(train_loss)
    best_index = min(range(count), key=lambda i: validation_loss[i])
    gaps = [validation_loss[i] - train_loss[i] for i in range(count)]
    if count > 1:
        mean_x = (count - 1) / 2
        mean_gap = sum(gaps) / count
        slope = sum((i - mean_x) * (gaps[i] - mean_gap) for i in range(count)) / sum(
            (i - mean_x) ** 2 for i in range(count)
        )
    else:
        slope = 0.0
    since_best = count - 1 - best_index
    worse_after_best = since_best > 0 and train_loss[-1] < train_loss[best_index]
    return {
        "diag.best_epoch": float(best_index + 1),
        "diag.best_validation_loss": float(validation_loss[best_index]),
        "diag.epochs_since_best": float(since_best),
        "diag.final_gap": float(gaps[-1]),
        "diag.gap_trend": float(slope),
        "diag.overfit_signal": 1.0 if worse_after_best else 0.0,
    }


class ExperimentRun:
    """Small explicit-run adapter; never changes MLflow's global active run."""

    def __init__(self, client: Any = None, run_id: str | None = None, entities: Any = None) -> None:
        self._client = client
        self._entities = entities
        self.run_id = run_id
        self.store_uri: str | None = None

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._client is not None:
            values = [
                self._entities.Param(key, str(value))
                for key, value in params.items()
                if value is not None
            ]
            if values:
                self._client.log_batch(self.run_id, params=values, synchronous=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int = 0) -> None:
        if self._client is not None:
            for key, value in metrics.items():
                if not math.isfinite(value):
                    raise ValueError(f"Cannot track non-finite metric: {key}")
            timestamp = int(time.time() * 1000)
            values = [
                self._entities.Metric(key, float(value), timestamp, step)
                for key, value in metrics.items()
            ]
            if values:
                self._client.log_batch(self.run_id, metrics=values, synchronous=True)

    def log_epoch(
        self,
        epoch: int,
        *,
        train_loss: float,
        validation_loss: float,
        learning_rate: float | None = None,
    ) -> None:
        """Record one epoch so versions can be graphed against each other."""
        metrics = {
            "train.loss": train_loss,
            "validation.loss": validation_loss,
            "train.val_gap": validation_loss - train_loss,
        }
        if learning_rate is not None:
            metrics["train.learning_rate"] = learning_rate
        self.log_metrics(metrics, step=epoch)

    def log_training_summary(
        self, train_loss: Sequence[float], validation_loss: Sequence[float]
    ) -> None:
        """Record epoch counts and the overfitting indicators for a finished curve."""
        self.log_params({"epochs_completed": len(train_loss)})
        self.log_metrics(training_diagnostics(train_loss, validation_loss))

    def log_dict(self, data: Mapping[str, Any], filename: str) -> None:
        if self._client is not None:
            self._client.log_dict(self.run_id, dict(data), filename)

    def log_figure(self, figure: Any, filename: str) -> None:
        if self._client is not None:
            self._client.log_figure(self.run_id, figure, filename)


@contextmanager
def track_experiment(
    config: MLflowConfig | None,
    *,
    kind: str,
    repository: Path,
    historical: bool = False,
    source_revision: str | None = None,
) -> Iterator[ExperimentRun]:
    """Track an explicitly enabled run and mark exceptions or interruptions failed."""
    if config is None:
        yield ExperimentRun()
        return

    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    try:
        client_type = importlib.import_module("mlflow.tracking").MlflowClient
        entities = importlib.import_module("mlflow.entities")
    except ImportError as exc:
        raise MissingDependencyError("MLflow experiment tracking", "tracking", "mlflow") from exc

    uri = config.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    artifact_location = None
    if not uri:
        root = repository.resolve() / ".mlflow"
        root.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{root / 'mlflow.db'}"
        artifact_location = (root / "artifacts").as_uri()
    name = (
        config.experiment_name
        or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        or "failurelens-development"
    )
    client = client_type(tracking_uri=uri)
    experiment = client.get_experiment_by_name(name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(name, artifact_location=artifact_location)
    )
    git_status = _git(repository, "status", "--porcelain")
    tags = {
        "failurelens.kind": kind,
        "mlflow.source.git.commit": source_revision
        or ("unknown" if historical else _git(repository, "rev-parse", "HEAD") or "unknown"),
        "failurelens.git_dirty": "unknown"
        if historical or git_status == "unknown"
        else str(bool(git_status)).lower(),
    }
    if historical:
        tags["failurelens.historical"] = "true"
    if config.model_version:
        tags["model.version"] = config.model_version
    run = client.create_run(experiment_id, tags=tags, run_name=config.run_name)
    tracker = ExperimentRun(client, run.info.run_id, entities)
    tracker.store_uri = uri
    started = time.perf_counter()
    try:
        if config.model_version:
            tracker.log_params({"model.version": config.model_version})
        tracker.log_dict(
            _environment(), "import_environment.json" if historical else "environment.json"
        )
        yield tracker
        duration_key = "import.duration_seconds" if historical else "run.duration_seconds"
        tracker.log_metrics({duration_key: time.perf_counter() - started})
    except BaseException:
        try:
            client.set_terminated(tracker.run_id, status="FAILED")
        except Exception:
            logging.getLogger(__name__).warning(
                "Could not mark MLflow run %s failed", tracker.run_id
            )
        raise
    else:
        client.set_terminated(tracker.run_id, status="FINISHED")
