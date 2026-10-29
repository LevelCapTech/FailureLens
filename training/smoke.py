"""Offline deterministic training smoke path for the toy pipeline."""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from training.config import TrainConfig
from training.data import load_training_examples
from training.objective import InfoNCEObjective, ToyEncoder


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _serialize_weights(weights: list[list[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in weights]


def _checkpoint_file(path: Path, epoch: int) -> Path:
    return path / f"checkpoint_epoch_{epoch}.json"


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    encoder: ToyEncoder,
    history: list[float],
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "history": history,
        "weights": _serialize_weights(encoder.weights),
        "config": config.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_checkpoint(path: Path) -> tuple[ToyEncoder, list[float], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoder = ToyEncoder(dim=len(payload["weights"]), seed=0)
    encoder.weights = [[float(value) for value in row] for row in payload["weights"]]
    history = [float(value) for value in payload.get("history", [])]
    epoch = int(payload.get("epoch", 0))
    return encoder, history, epoch


def _metadata(config: TrainConfig, history: list[float], dataset_version: str) -> dict[str, Any]:
    return {
        "git_sha": _git_sha(),
        "config": config.model_dump(mode="json"),
        "dataset_version": dataset_version,
        "dataset": config.dataset,
        "objective": config.objective,
        "seed": config.seed,
        "revision": config.revision,
        "dependencies": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "random_state": random.getstate()[1][0] if random.getstate()[1] else 0,
        },
        "hardware": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "history": history,
        "metrics": {
            "final_loss": history[-1] if history else float("nan"),
        },
    }


def run_toy_training(
    config: TrainConfig | str | Path | None = None,
    *,
    dataset_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a tiny offline training loop and export artifacts."""
    config_value = (
        config
        if isinstance(config, TrainConfig)
        else TrainConfig.from_json(config or "training/configs/toy_config.json")
    )
    output_root = Path(output_dir) if output_dir is not None else Path(config_value.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    examples = load_training_examples(dataset_path)
    if not examples:
        raise ValueError("training dataset is empty")

    objective = InfoNCEObjective()
    if resume_from is not None:
        checkpoint_path = Path(resume_from)
        encoder, history, start_epoch = _load_checkpoint(checkpoint_path)
    else:
        encoder = ToyEncoder(dim=8, seed=config_value.seed)
        history = []
        start_epoch = 0

    if resume_from is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

    total_batches = len(examples)
    for epoch in range(start_epoch + 1, config_value.epochs + 1):
        epoch_loss = 0.0
        for example in examples[:total_batches]:
            loss, gradients = objective.loss_and_gradients(
                encoder,
                example.query,
                example.positive,
                example.negative,
            )
            encoder.apply_gradients(gradients, config_value.learning_rate)
            epoch_loss += loss
        history.append(epoch_loss / max(1, len(examples)))
        _save_checkpoint(
            _checkpoint_file(output_root, epoch),
            epoch=epoch,
            encoder=encoder,
            history=history,
            config=config_value,
        )

    artifact = {
        "config": config_value.model_dump(mode="json"),
        "dataset_version": examples[0].dataset_version,
        "source_repository": examples[0].source_repository,
        "history": history,
        "metrics": {"final_loss": history[-1] if history else 0.0},
        "checkpoint": str(_checkpoint_file(output_root, config_value.epochs)),
        "metadata": _metadata(config_value, history, examples[0].dataset_version),
    }
    artifact_path = output_root / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the toy CPU training smoke test")
    parser.add_argument("--config", default="training/configs/toy_config.json")
    parser.add_argument("--dataset", default="datasets/fixtures/sample_record.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    run_toy_training(
        args.config,
        dataset_path=args.dataset,
        resume_from=args.resume_from,
        output_dir=args.output_dir,
    )
