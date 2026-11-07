"""Import recorded Stage 2 losses without retraining or uploading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from failurelens.experiment_tracking import (
    MLflowConfig,
    add_tracking_arguments,
    track_experiment,
    tracking_from_args,
)


class TrainingHistory(BaseModel):
    """Validate measurements before creating any tracking run."""

    model_config = ConfigDict(extra="forbid")

    train_loss: list[FiniteFloat] = Field(min_length=1)
    val_loss: list[FiniteFloat] = Field(min_length=1)

    @model_validator(mode="after")
    def matching_epochs(self) -> TrainingHistory:
        if len(self.train_loss) != len(self.val_loss):
            raise ValueError("training and validation histories must cover the same epochs")
        return self


class RecordedTraining(BaseModel):
    """Only measured fields from an existing experiment are imported."""

    model_config = ConfigDict(extra="forbid")

    history: TrainingHistory
    git_sha: str = Field(pattern=r"^[a-f0-9]{40}$")
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    n_train: int = Field(gt=0)
    n_val: int = Field(gt=0)
    elapsed_seconds: FiniteFloat = Field(ge=0)
    peak_gpu_memory_mib: int = Field(ge=0)


def import_history(path: Path, *, tracking: MLflowConfig) -> str:
    """Import local experiment metadata or a saved public cloud provenance JSON."""
    raw = path.read_bytes()
    document = json.loads(raw)
    source = (
        document["training"] if document.get("build_kind") == "hf-jobs-stage2-rebuild" else document
    )
    recorded = RecordedTraining.model_validate(
        {key: source[key] for key in RecordedTraining.model_fields}
    )
    if "training" in document and document["artifact_sha256"] != recorded.artifact_sha256:
        raise ValueError("cloud provenance and training artifact digests do not match")
    if source["config"]["epochs"] != len(recorded.history.train_loss):
        raise ValueError("recorded epoch count does not match the completed history")
    config_keys = (
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "temperature",
        "optimizer",
        "objective",
    )
    parameters = {key: source["config"][key] for key in config_keys if key in source["config"]}
    metadata = source.get("run_metadata", {})
    for key in ("model_id", "model_revision", "dataset_sha256", "trainable_params", "total_params"):
        if key in metadata:
            parameters[key] = metadata[key]
    parameters.update(
        {
            "n_train": recorded.n_train,
            "n_val": recorded.n_val,
            "artifact_sha256": recorded.artifact_sha256,
            "history_source_sha256": hashlib.sha256(raw).hexdigest(),
            "measurement_scope": "historical-training-only",
        }
    )
    environment = {
        key: metadata[key]
        for key in ("python", "torch_version", "device_name", "device_memory_mib")
        if key in metadata
    }
    if "dependencies" in document:
        environment["dependencies"] = {
            key: document["dependencies"][key]
            for key in (
                "torch",
                "transformers",
                "sentence-transformers",
                "huggingface_hub",
                "safetensors",
                "einops",
                "pydantic",
                "jsonschema",
                "typer",
            )
            if key in document["dependencies"]
        }

    with track_experiment(
        tracking,
        kind="stage2-training",
        repository=Path(__file__).resolve().parents[2],
        historical=True,
        source_revision=recorded.git_sha,
    ) as experiment:
        experiment.log_params(parameters)
        experiment.log_dict(parameters, "config.json")
        experiment.log_dict(environment, "training_environment.json")
        for epoch, (train_loss, val_loss) in enumerate(
            zip(recorded.history.train_loss, recorded.history.val_loss, strict=True), 1
        ):
            experiment.log_metrics(
                {"train.loss": train_loss, "validation.loss": val_loss}, step=epoch
            )
        experiment.log_metrics(
            {
                "train.elapsed_seconds": recorded.elapsed_seconds,
                "train.peak_gpu_memory_mib": recorded.peak_gpu_memory_mib,
            }
        )
        experiment.log_dict(
            {
                "history": recorded.history.model_dump(),
                "artifact_sha256": recorded.artifact_sha256,
                "measurement_scope": "historical-training-only",
            },
            "training.json",
        )

        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 4), layout="constrained")
        axes = figure.subplots()
        epochs = list(range(1, len(recorded.history.train_loss) + 1))
        axes.plot(epochs, recorded.history.train_loss, "o-", label="Training")
        axes.plot(epochs, recorded.history.val_loss, "o-", label="Validation")
        axes.set(
            xlabel="Epoch",
            ylabel="InfoNCE loss",
            title="Recorded Stage 2 training (historical)",
            xticks=epochs,
            ylim=(0, None),
        )
        axes.legend()
        axes.grid(alpha=0.2)
        experiment.log_figure(figure, "loss_curves.png")
        assert experiment.run_id is not None
        return experiment.run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metadata", type=Path, help="An existing experiment_metadata.json or cloud provenance JSON."
    )
    add_tracking_arguments(parser)
    args = parser.parse_args(argv)
    tracking = tracking_from_args(parser, args)
    if tracking is None:
        parser.error("Importing historical measurements requires --mlflow")
    print(import_history(args.metadata, tracking=tracking))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
