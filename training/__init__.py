"""Minimal training utilities for the CPU-only toy pipeline."""

from __future__ import annotations

from training.config import TrainConfig
from training.data import TrainingExample, load_training_examples
from training.objective import InfoNCEObjective, ToyEncoder
from training.smoke import run_toy_training

__all__ = [
    "InfoNCEObjective",
    "ToyEncoder",
    "TrainConfig",
    "TrainingExample",
    "load_training_examples",
    "run_toy_training",
]
