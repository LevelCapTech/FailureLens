"""Opt-in dependency tests using real MLflow SQLite stores and artifact reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from evaluation.run_baseline_eval import DEFAULT_BENCHMARK, run
from training.config import TrainConfig
from training.scripts.import_history import import_history
from training.smoke import run_toy_training

from failurelens.experiment_tracking import MLflowConfig, track_experiment


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, MLflowConfig]:
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    mlflow = pytest.importorskip("mlflow")
    config = MLflowConfig(
        tracking_uri=f"sqlite:///{tmp_path / 'experiments.db'}", experiment_name="test"
    )
    client = mlflow.MlflowClient(tracking_uri=config.tracking_uri)
    client.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    return client, config


def _runs(client: Any) -> list[Any]:
    experiment = client.get_experiment_by_name("test")
    return list(client.search_runs([experiment.experiment_id]))


def test_baseline_real_metrics_match_saved_report_and_identity(
    store: tuple[Any, MLflowConfig],
) -> None:
    client, config = store
    report = run(DEFAULT_BENCHMARK, tracking=config)
    tracked = _runs(client)[0]
    assert tracked.info.status == "FINISHED"
    assert tracked.data.tags["failurelens.kind"] == "baseline-evaluation"
    assert (
        tracked.data.params["benchmark_sha256"]
        == hashlib.sha256(DEFAULT_BENCHMARK.read_bytes()).hexdigest()
    )
    assert tracked.data.params["analysis_method"] == report["analysis_method"]
    assert tracked.data.params["score_type"] == "heuristic"
    for level in ("file_level", "function_level", "performance"):
        for key, value in report[level].items():
            assert tracked.data.metrics[f"{level}.{key}"] == value
    artifact = Path(client.download_artifacts(tracked.info.run_id, "evaluation.json"))
    assert json.loads(artifact.read_text()) == report
    assert {item.path for item in client.list_artifacts(tracked.info.run_id)} == {
        "environment.json",
        "evaluation.json",
    }


def test_toy_loss_steps_and_resume_are_real(
    store: tuple[Any, MLflowConfig], tmp_path: Path
) -> None:
    client, config = store
    cfg = TrainConfig.from_json("training/configs/toy_config.json")
    artifact = run_toy_training(cfg, output_dir=tmp_path / "first", tracking=config)
    tracked = _runs(client)[0]
    history = client.get_metric_history(tracked.info.run_id, "train.loss")
    assert [(entry.step, entry.value) for entry in history] == list(
        enumerate(artifact["history"], 1)
    )
    assert tracked.data.params["optimizer"] == "toy-gradient-descent"
    assert tracked.data.metrics["train.final_loss"] == artifact["history"][-1]
    assert "output_dir" not in tracked.data.params
    assert {item.path for item in client.list_artifacts(tracked.info.run_id)} == {
        "environment.json",
        "config.json",
        "training.json",
    }

    resumed = run_toy_training(
        cfg.model_copy(update={"epochs": cfg.epochs + 1}),
        output_dir=tmp_path / "resumed",
        resume_from=artifact["checkpoint"],
        tracking=config,
    )
    resumed_run = next(item for item in _runs(client) if item.info.run_id != tracked.info.run_id)
    continued = client.get_metric_history(resumed_run.info.run_id, "train.loss")
    assert [(entry.step, entry.value) for entry in continued] == [
        (cfg.epochs + 1, resumed["history"][-1])
    ]
    assert resumed_run.data.params["resume_epoch"] == str(cfg.epochs)
    assert resumed_run.info.status == "FINISHED"


def test_invalid_benchmark_records_failed_run(
    store: tuple[Any, MLflowConfig], tmp_path: Path
) -> None:
    client, config = store
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not JSON", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        run(invalid, tracking=config)
    assert _runs(client)[0].info.status == "FAILED"


def test_existing_fluent_run_and_tracking_uri_are_untouched(
    store: tuple[Any, MLflowConfig], tmp_path: Path
) -> None:
    import mlflow

    client, config = store
    previous_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(config.tracking_uri)
        experiment = client.get_experiment_by_name("test")
        with mlflow.start_run(experiment_id=experiment.experiment_id) as parent:
            with track_experiment(config, kind="test", repository=tmp_path) as child:
                assert child.run_id != parent.info.run_id
                assert mlflow.active_run().info.run_id == parent.info.run_id
            assert mlflow.active_run().info.run_id == parent.info.run_id
            assert mlflow.get_tracking_uri() == config.tracking_uri
    finally:
        mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("cloud", [False, True])
def test_historical_import_preserves_measurements_and_filters_private_paths(
    store: tuple[Any, MLflowConfig], tmp_path: Path, cloud: bool
) -> None:
    client, config = store
    record = {
        "history": {"train_loss": [0.7, 0.5], "val_loss": [0.8, 0.7]},
        "config": {"epochs": 2, "seed": 42, "private_path": "/private-fixture-path"},
        "git_sha": "b" * 40,
        "artifact_sha256": "a" * 64,
        "artifact_path": "/private-fixture-path/model.safetensors",
        "n_train": 6,
        "n_val": 6,
        "elapsed_seconds": 3.5,
        "peak_gpu_memory_mib": 885,
        "run_metadata": {"device_name": "test-GPU", "model_revision": "c" * 40},
    }
    document = (
        {"build_kind": "hf-jobs-stage2-rebuild", "training": record, "artifact_sha256": "a" * 64}
        if cloud
        else record
    )
    path = tmp_path / "record.json"
    path.write_text(json.dumps(document))
    run_id = import_history(path, tracking=config)
    tracked = client.get_run(run_id)
    assert tracked.info.status == "FINISHED"
    assert tracked.data.tags["failurelens.historical"] == "true"
    assert tracked.data.tags["mlflow.source.git.commit"] == record["git_sha"]
    assert tracked.data.tags["failurelens.git_dirty"] == "unknown"
    assert tracked.data.params["artifact_sha256"] == record["artifact_sha256"]
    assert tracked.data.params["measurement_scope"] == "historical-training-only"
    assert tracked.data.metrics["train.elapsed_seconds"] == record["elapsed_seconds"]
    assert "run.duration_seconds" not in tracked.data.metrics
    assert tracked.data.metrics["import.duration_seconds"] >= 0
    for metric, key in (("train.loss", "train_loss"), ("validation.loss", "val_loss")):
        points = client.get_metric_history(run_id, metric)
        assert [(point.step, point.value) for point in points] == list(
            enumerate(record["history"][key], 1)
        )
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert artifacts == {
        "import_environment.json",
        "training_environment.json",
        "training.json",
        "config.json",
        "loss_curves.png",
    }
    for artifact in artifacts:
        downloaded = Path(client.download_artifacts(run_id, artifact))
        if artifact.endswith(".json"):
            assert "/private-fixture-path" not in downloaded.read_text()
        else:
            assert downloaded.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
