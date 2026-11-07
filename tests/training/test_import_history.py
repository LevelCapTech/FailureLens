"""Historical imports reject invalid measurements before opening MLflow runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from training.scripts.import_history import import_history

from failurelens.experiment_tracking import MLflowConfig


@pytest.fixture
def recorded_training() -> dict[str, Any]:
    return {
        "history": {"train_loss": [0.7, 0.5], "val_loss": [0.8, 0.7]},
        "config": {"epochs": 2},
        "git_sha": "b" * 40,
        "artifact_sha256": "a" * 64,
        "n_train": 6,
        "n_val": 6,
        "elapsed_seconds": 3.5,
        "peak_gpu_memory_mib": 885,
    }


@pytest.mark.parametrize(
    "invalid", ["nan", "missing_epoch", "epoch_count", "empty", "artifact_digest", "no_validation"]
)
def test_reject_invalid_measurements(
    recorded_training: dict[str, Any], tmp_path: Path, invalid: str
) -> None:
    record = recorded_training
    if invalid == "nan":
        record["history"]["train_loss"][0] = float("nan")
    elif invalid == "missing_epoch":
        record["history"]["val_loss"].pop()
    elif invalid == "epoch_count":
        record["config"]["epochs"] = 3
    elif invalid == "empty":
        record["history"] = {"train_loss": [], "val_loss": []}
    elif invalid == "artifact_digest":
        record["artifact_sha256"] = "not-a-digest"
    elif invalid == "no_validation":
        record["n_val"] = 0
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        import_history(path, tracking=MLflowConfig())


def test_reject_cloud_weight_identity_mismatch(
    recorded_training: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(
        json.dumps(
            {
                "build_kind": "hf-jobs-stage2-rebuild",
                "training": recorded_training,
                "artifact_sha256": "c" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="artifact digests do not match"):
        import_history(path, tracking=MLflowConfig())
