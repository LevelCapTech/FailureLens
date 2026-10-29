from __future__ import annotations

import json
from pathlib import Path

import pytest
from training.config import TrainConfig
from training.data import load_training_examples
from training.objective import InfoNCEObjective, ToyEncoder
from training.smoke import run_toy_training


def test_train_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        TrainConfig.model_validate(
            {
                "base_model": "",
                "revision": "fixed-sha",
                "dataset": "fixture",
                "dataset_version": "v1",
                "seed": 0,
                "optimizer": "adamw",
                "learning_rate": 0.1,
                "batch_size": 2,
                "epochs": 1,
                "max_length": 16,
                "objective": "infonce",
                "output_dir": "tmp",
            }
        )

    with pytest.raises(ValueError):
        TrainConfig.model_validate(
            {
                "base_model": "toy",
                "revision": "fixed-sha",
                "dataset": "fixture",
                "dataset_version": "v1",
                "seed": 0,
                "optimizer": "adamw",
                "learning_rate": 0.1,
                "batch_size": 0,
                "epochs": 1,
                "max_length": 16,
                "objective": "infonce",
                "output_dir": "tmp",
            }
        )


def test_load_training_examples_rejects_bad_quality_and_excluded_sources() -> None:
    examples = load_training_examples(Path("datasets/fixtures/sample_record.json"))
    assert len(examples) == 1
    assert examples[0].source_repository == "examples/fixtures/sample_project"

    bad_path = Path("/tmp/failurelens_training_bad.json")
    record = json.loads(Path("datasets/fixtures/sample_record.json").read_text())
    record["source_repository"] = "psf/requests"
    bad_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="excluded"):
        load_training_examples(bad_path)
    bad_path.unlink(missing_ok=True)

    broken = json.loads(Path("datasets/fixtures/sample_record.json").read_text())
    broken["error_log"] = None
    broken_path = Path("/tmp/failurelens_training_broken.json")
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="error_log"):
        load_training_examples(broken_path)
    broken_path.unlink(missing_ok=True)


def test_infonce_objective_backward_pass_reduces_loss() -> None:
    encoder = ToyEncoder(dim=4, seed=1)
    objective = InfoNCEObjective()
    query = "ZeroDivisionError division by zero in average"
    positive = "average returns divide with empty values"
    negative = "database connection timeout"
    before = objective.loss_and_gradients(encoder, query, positive, negative)[0]
    _, gradients = objective.loss_and_gradients(encoder, query, positive, negative)
    encoder.apply_gradients(gradients, 0.05)
    after = objective.loss_and_gradients(encoder, query, positive, negative)[0]
    assert after < before


def test_checkpoint_resume_and_deterministic_smoke() -> None:
    cfg = TrainConfig.model_validate(
        {
            "base_model": "toy-bi-encoder",
            "revision": "train-toy-0.1.0",
            "dataset": "datasets/fixtures/sample_record.json",
            "dataset_version": "fixture-v1",
            "seed": 7,
            "optimizer": "adamw",
            "learning_rate": 0.05,
            "batch_size": 1,
            "epochs": 2,
            "max_length": 32,
            "objective": "infonce",
            "mixed_precision": False,
            "output_dir": "/tmp/failurelens-toy-run",
            "val_interval": 1,
            "early_stopping": False,
        }
    )
    first = run_toy_training(cfg, dataset_path="datasets/fixtures/sample_record.json")
    second = run_toy_training(cfg, dataset_path="datasets/fixtures/sample_record.json")
    assert first["history"] == second["history"]
    assert Path(first["checkpoint"]).exists()

    checkpoint = Path(first["checkpoint"])
    resumed = run_toy_training(
        cfg,
        dataset_path="datasets/fixtures/sample_record.json",
        resume_from=checkpoint,
        output_dir="/tmp/failurelens-toy-resume",
    )
    assert resumed["metadata"]["git_sha"]
