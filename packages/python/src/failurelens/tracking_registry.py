"""MLflow Model Registry helpers for FailureLens model versions (development only).

Registered model name: ``failurelens``. Each version carries honest tags: the
source commit, the weight SHA-256, the model label (``failurelens.version``),
and an evaluation status that starts as ``not evaluated``. Measured aggregate
scores are attached later with :func:`record_evaluation`.

Weights are copied only into a local file-based store. For a remote or
non-file store only the digest and a hash of the local path are recorded.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from failurelens.errors import MissingDependencyError
from failurelens.experiment_tracking import MODEL_VERSION_PATTERN

REGISTERED_MODEL_NAME = "failurelens"
EVALUATION_STATUSES = (
    "not evaluated",
    "gate-c-pending",
    "evaluated-development",
    "gate-c-evaluated",
)
MODEL_STATUSES = ("experimental", "tainted-do-not-use")
_METRIC_KEY = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMMIT = re.compile(r"^([a-f0-9]{40}|unknown)$")
_VERSION_TAG = "failurelens.version"


def _client(tracking_uri: str | None, repository: Path | None) -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    try:
        client_type = importlib.import_module("mlflow.tracking").MlflowClient
    except ImportError as exc:
        raise MissingDependencyError("MLflow model registry", "tracking", "mlflow") from exc
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        root = (repository or Path.cwd()).resolve() / ".mlflow"
        uri = f"sqlite:///{root / 'mlflow.db'}"
    return client_type(tracking_uri=uri, registry_uri=uri), uri


def is_local_store(tracking_uri: str, artifact_uri: str) -> bool:
    """True only for file/SQLite tracking whose artifacts are on the local disk."""
    tracking_local = urlsplit(tracking_uri).scheme in ("", "sqlite", "file")
    artifact_scheme = urlsplit(artifact_uri).scheme
    return tracking_local and artifact_scheme in ("", "file")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_model_version(
    *,
    run_id: str,
    model_version: str,
    source_commit: str,
    weight_sha256: str,
    weights_path: Path | None = None,
    evaluation_status: str = "not evaluated",
    status: str = "experimental",
    historical: bool = False,
    tracking_uri: str | None = None,
    repository: Path | None = None,
    copy_weights: bool = True,
    extra_tags: Mapping[str, str] | None = None,
) -> str:
    """Register a trained checkpoint of ``run_id`` and return the registry version number."""
    if not re.match(MODEL_VERSION_PATTERN, model_version):
        raise ValueError("model_version must look like v0.2.0 or v0.2.0-candidate-1")
    if not _SHA256.match(weight_sha256):
        raise ValueError("weight_sha256 must be a lowercase SHA-256 hex digest")
    if not _COMMIT.match(source_commit):
        raise ValueError("source_commit must be a 40-hex commit or 'unknown'")
    if evaluation_status not in EVALUATION_STATUSES:
        raise ValueError(f"evaluation_status must be one of {EVALUATION_STATUSES}")
    if status not in MODEL_STATUSES:
        raise ValueError(f"status must be one of {MODEL_STATUSES}")
    if weights_path is not None and _sha256_file(weights_path) != weight_sha256:
        raise ValueError("weights file does not match weight_sha256")

    client, uri = _client(tracking_uri, repository)
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    upload = bool(copy_weights and weights_path is not None and is_local_store(uri, artifact_uri))
    metadata: dict[str, Any] = {
        "model_version": model_version,
        "weight_sha256": weight_sha256,
        "source_commit": source_commit,
        "weights_copied_to_store": upload,
    }
    if weights_path is not None:
        metadata["weight_size_bytes"] = weights_path.stat().st_size
        metadata["local_path_sha256"] = hashlib.sha256(
            str(weights_path.resolve()).encode()
        ).hexdigest()
    client.log_dict(run_id, metadata, "model_metadata/weights.json")
    if upload:
        assert weights_path is not None
        client.log_artifact(run_id, str(weights_path), artifact_path="model")
        source = f"{artifact_uri}/model"
    else:
        source = f"{artifact_uri}/model_metadata"

    if not _registered_model_exists(client):
        client.create_registered_model(
            REGISTERED_MODEL_NAME,
            description="FailureLens development model versions. Not a quality claim.",
        )
    tags = {
        _VERSION_TAG: model_version,
        "source_commit": source_commit,
        "weight_sha256": weight_sha256,
        "evaluation_status": evaluation_status,
        "status": status,
        "historical": str(historical).lower(),
        "weights_in_store": str(upload).lower(),
    }
    tags.update(extra_tags or {})
    version = client.create_model_version(
        REGISTERED_MODEL_NAME, source=source, run_id=run_id, tags=tags
    )
    return str(version.version)


def _registered_model_exists(client: Any) -> bool:
    return bool(client.search_registered_models(filter_string=f"name = '{REGISTERED_MODEL_NAME}'"))


def _find_version(client: Any, model_version: str) -> Any:
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    matches = [
        v
        for v in versions
        if v.tags.get(_VERSION_TAG) == model_version or str(v.version) == model_version
    ]
    if not matches:
        raise LookupError(f"No registered {REGISTERED_MODEL_NAME} version matches {model_version}")
    return max(matches, key=lambda v: int(v.version))


def record_evaluation(
    model_version: str,
    metrics: Mapping[str, float],
    *,
    status: str = "evaluated-development",
    tracking_uri: str | None = None,
    repository: Path | None = None,
) -> str:
    """Attach measured aggregate scores to an existing registry version.

    ``model_version`` is a label such as ``v0.1.0`` or a registry version
    number. Scores become ``eval.<name>`` tags on the version and ``eval.<name>``
    metrics on its source run. Returns the registry version number.
    """
    if status not in EVALUATION_STATUSES or status == "not evaluated":
        raise ValueError(f"status must be one of {EVALUATION_STATUSES[1:]}")
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if not _METRIC_KEY.match(key):
            raise ValueError(f"Unsafe metric name: {key!r}")
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"Cannot record non-finite or boolean metric: {key}")
        clean[key] = float(value)
    client, _ = _client(tracking_uri, repository)
    version = _find_version(client, model_version)
    number = str(version.version)
    for key, value in clean.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, number, f"eval.{key}", f"{value:.6g}")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, number, "evaluation_status", status)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME,
        number,
        "evaluation_updated_at",
        datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    if version.run_id and clean:
        from mlflow.entities import Metric

        stamp = int(datetime.now(UTC).timestamp() * 1000)
        client.log_batch(
            version.run_id,
            metrics=[Metric(f"eval.{key}", value, stamp, 0) for key, value in clean.items()],
            synchronous=True,
        )
    return number
