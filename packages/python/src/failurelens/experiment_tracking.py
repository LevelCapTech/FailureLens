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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from failurelens.errors import MissingDependencyError


class MLflowConfig(BaseModel):
    """Supplying this config explicitly enables experiment tracking."""

    model_config = ConfigDict(extra="forbid")

    tracking_uri: str | None = None
    experiment_name: str | None = Field(default=None, min_length=1)
    run_name: str | None = Field(default=None, min_length=1)


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same opt-in flags to evaluation and training entry points."""
    group = parser.add_argument_group("optional MLflow experiment tracking")
    group.add_argument("--mlflow", action="store_true", help="Record this experiment in MLflow.")
    group.add_argument(
        "--mlflow-tracking-uri", help="Override MLFLOW_TRACKING_URI (default: local SQLite)."
    )
    group.add_argument("--mlflow-experiment", help="Override MLFLOW_EXPERIMENT_NAME.")
    group.add_argument("--mlflow-run-name", help="Optional display name for this run.")


def tracking_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> MLflowConfig | None:
    """Require explicit opt-in even when MLflow environment variables are set."""
    if not args.mlflow:
        if args.mlflow_tracking_uri or args.mlflow_experiment or args.mlflow_run_name:
            parser.error("MLflow options require --mlflow")
        return None
    return MLflowConfig(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=args.mlflow_run_name,
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


class ExperimentRun:
    """Small explicit-run adapter; never changes MLflow's global active run."""

    def __init__(self, client: Any = None, run_id: str | None = None, entities: Any = None) -> None:
        self._client = client
        self._entities = entities
        self.run_id = run_id

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
    run = client.create_run(experiment_id, tags=tags, run_name=config.run_name)
    tracker = ExperimentRun(client, run.info.run_id, entities)
    started = time.perf_counter()
    try:
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
