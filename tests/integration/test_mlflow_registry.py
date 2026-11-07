"""Real MLflow SQLite stores: per-epoch gap, version tags, registry and evaluation updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from evaluation.run_baseline_eval import DEFAULT_BENCHMARK, run
from training.scripts.import_history import import_history

from failurelens.experiment_tracking import MLflowConfig
from failurelens.tracking_registry import (
    REGISTERED_MODEL_NAME,
    is_local_store,
    record_evaluation,
    register_model_version,
)

pytest.importorskip("mlflow")
import mlflow


@pytest.fixture
def uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    store = f"sqlite:///{tmp_path / 'm.db'}"
    mlflow.MlflowClient(tracking_uri=store).create_experiment(
        "test", artifact_location=(tmp_path / "art").as_uri()
    )
    return store


def _record(tmp_path: Path, name: str = "rec.json") -> Path:
    record = {
        "history": {"train_loss": [0.8, 0.5, 0.3], "val_loss": [0.7, 0.6, 0.9]},
        "config": {"epochs": 3, "learning_rate": 2e-5, "batch_size": 2, "seed": 42},
        "git_sha": "c" * 40,
        "artifact_sha256": "d" * 64,
        "n_train": 6,
        "n_val": 6,
        "elapsed_seconds": 3.5,
        "peak_gpu_memory_mib": 885,
    }
    path = tmp_path / name
    path.write_text(json.dumps(record))
    return path


def test_history_import_logs_gap_lr_diagnostics_and_registers(uri: str, tmp_path: Path) -> None:
    config = MLflowConfig(tracking_uri=uri, experiment_name="test", model_version="v0.1.0")
    run_id = import_history(_record(tmp_path), tracking=config)
    client = mlflow.MlflowClient(tracking_uri=uri)
    data = client.get_run(run_id).data
    assert data.tags["model.version"] == "v0.1.0"
    assert data.params["epochs_total"] == "3"
    assert data.params["epochs_completed"] == "3"
    assert data.params["learning_rate_schedule"].endswith("not-measured")
    gap = [m.value for m in client.get_metric_history(run_id, "train.val_gap")]
    assert gap == pytest.approx([-0.1, 0.1, 0.6])
    lrs = [m.value for m in client.get_metric_history(run_id, "train.learning_rate")]
    assert lrs[0] == pytest.approx(2e-5 * 0.75) and lrs[-1] == pytest.approx(0.0, abs=1e-12)
    assert data.metrics["diag.overfit_signal"] == 1.0

    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1
    tags = versions[0].tags
    assert tags["failurelens.version"] == "v0.1.0"
    assert tags["source_commit"] == "c" * 40
    assert tags["weight_sha256"] == "d" * 64
    assert tags["evaluation_status"] == "not evaluated"
    assert tags["historical"] == "true"
    assert tags["weights_in_store"] == "false"


def test_weights_copied_locally_and_digest_checked(uri: str, tmp_path: Path) -> None:
    run_id = import_history(
        _record(tmp_path), tracking=MLflowConfig(tracking_uri=uri, experiment_name="test")
    )
    weights = tmp_path / "w.safetensors"
    weights.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    with pytest.raises(ValueError, match="does not match"):
        register_model_version(
            run_id=run_id,
            model_version="v0.2.0-candidate-1",
            source_commit="e" * 40,
            weight_sha256="0" * 64,
            weights_path=weights,
            tracking_uri=uri,
        )
    number = register_model_version(
        run_id=run_id,
        model_version="v0.2.0-candidate-1",
        source_commit="e" * 40,
        weight_sha256=digest,
        weights_path=weights,
        tracking_uri=uri,
    )
    client = mlflow.MlflowClient(tracking_uri=uri)
    version = client.get_model_version(REGISTERED_MODEL_NAME, number)
    assert version.tags["weights_in_store"] == "true"
    assert {a.path for a in client.list_artifacts(run_id, "model")} == {"model/w.safetensors"}
    meta_path = Path(client.download_artifacts(run_id, "model_metadata/weights.json"))
    meta = json.loads(meta_path.read_text())
    assert meta["weight_sha256"] == digest
    assert "local_path_sha256" in meta
    assert str(tmp_path) not in json.dumps(meta)


def test_non_local_store_never_uploads_weights() -> None:
    assert is_local_store("sqlite:////x/m.db", "file:///x/art")
    assert not is_local_store("https://mlflow.example.org", "file:///x/art")
    assert not is_local_store("sqlite:////x/m.db", "s3://bucket/art")
    assert not is_local_store("databricks", "dbfs:/x")


def test_record_evaluation_updates_same_version(uri: str, tmp_path: Path) -> None:
    config = MLflowConfig(tracking_uri=uri, experiment_name="test", model_version="v0.1.0")
    run_id = import_history(_record(tmp_path), tracking=config)
    number = record_evaluation(
        "v0.1.0",
        {"function_level.mrr": 0.5, "function_level.top_1_accuracy": 0.25},
        tracking_uri=uri,
    )
    client = mlflow.MlflowClient(tracking_uri=uri)
    tags = client.get_model_version(REGISTERED_MODEL_NAME, number).tags
    assert tags["eval.function_level.mrr"] == "0.5"
    assert tags["evaluation_status"] == "evaluated-development"
    assert client.get_run(run_id).data.metrics["eval.function_level.mrr"] == 0.5
    with pytest.raises(ValueError):
        record_evaluation("v0.1.0", {"x": float("nan")}, tracking_uri=uri)
    with pytest.raises(ValueError):
        record_evaluation("v0.1.0", {"bad key": 1.0}, tracking_uri=uri)
    with pytest.raises(LookupError):
        record_evaluation("v9.9.9", {"x": 1.0}, tracking_uri=uri)


def test_evaluation_records_model_version_and_dataset_digest(uri: str) -> None:
    config = MLflowConfig(
        tracking_uri=uri, experiment_name="test", model_version="v0.2.0-candidate-1"
    )
    run(DEFAULT_BENCHMARK, tracking=config)
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("test")
    assert experiment is not None
    tracked: Any = client.search_runs([experiment.experiment_id])[0]
    assert tracked.data.tags["model.version"] == "v0.2.0-candidate-1"
    assert tracked.data.params["dataset_sha256"] == tracked.data.params["benchmark_sha256"]


def test_compare_report_renders_versions_and_hides_tainted(uri: str, tmp_path: Path) -> None:
    from training.scripts.mlflow_compare import build_report

    import_history(
        _record(tmp_path, "a.json"),
        tracking=MLflowConfig(tracking_uri=uri, experiment_name="test", model_version="v0.1.0"),
    )
    import_history(
        _record(tmp_path, "b.json"),
        tracking=MLflowConfig(tracking_uri=uri, experiment_name="test", model_version="v0.0.9"),
        status="tainted-do-not-use",
    )
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    written = build_report(client, tmp_path / "reports")
    assert {p.name for p in written} == {
        "loss_curves.png",
        "generalization_gap.png",
        "learning_rate.png",
        "evaluation_scores.png",
        "index.html",
    }
    assert all(p.stat().st_size > 0 for p in written)
    page = (tmp_path / "reports" / "index.html").read_text()
    assert "v0.1.0" in page
    assert "v0.0.9" not in page
    assert "not evaluated" in page
    assert "warning" in page
    build_report(client, tmp_path / "all", include_tainted=True)
    assert "v0.0.9" in (tmp_path / "all" / "index.html").read_text()


def test_backfill_is_idempotent_and_retires_unlabelled_duplicate(uri: str, tmp_path: Path) -> None:
    from training.scripts.backfill_registry import Entry, backfill

    record = _record(tmp_path, "h.json")
    old = import_history(record, tracking=MLflowConfig(tracking_uri=uri, experiment_name="test"))
    entries = [
        Entry("v0.1.0", "v0.1.0 history", record, "experimental"),
        Entry("v0.0.1-tainted-v1", "tainted", tmp_path / "missing.json", "tainted-do-not-use"),
    ]
    first = backfill(entries, uri, "test", retire=True)
    assert any("registered v0.1.0" in m for m in first)
    assert any("record not found" in m for m in first)
    assert any("retired unlabelled duplicate" in m for m in first)
    client = mlflow.MlflowClient(tracking_uri=uri)
    assert client.get_run(old).info.lifecycle_stage == "deleted"
    second = backfill(entries, uri, "test", retire=True)
    assert any("already registered" in m for m in second)
    versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1
