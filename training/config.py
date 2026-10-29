"""Config schema and loader for the toy training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_EXCLUDED_REPOS: frozenset[str] = frozenset(
    {
        "psf/requests",
        "pallets/werkzeug",
        "pypa/pip",
    }
)


class TrainConfig(BaseModel):
    """Schema for the offline toy training pipeline."""

    model_config = ConfigDict(extra="forbid")

    base_model: str = Field(description="Base model identifier or toy model name.")
    revision: str = Field(description="Pinned model revision.")
    dataset: str = Field(description="Dataset name or fixture path.")
    dataset_version: str = Field(description="Version tag for the dataset.")
    seed: int = Field(default=0, ge=0)
    optimizer: str = Field(default="adamw")
    learning_rate: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=4, gt=0)
    epochs: int = Field(default=2, gt=0)
    max_length: int = Field(default=32, gt=0)
    objective: str = Field(default="infonce")
    mixed_precision: bool = Field(default=False)
    output_dir: str = Field(default="training/experiments/toy-smoke")
    val_interval: int = Field(default=1, gt=0)
    early_stopping: bool = Field(default=False)

    @model_validator(mode="after")
    def _validate_fields(self) -> TrainConfig:
        if not self.base_model.strip():
            raise ValueError("base_model must be a non-empty string")
        if not self.revision.strip():
            raise ValueError("revision must be a non-empty string")
        if not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        if not self.dataset_version.strip():
            raise ValueError("dataset_version must be a non-empty string")
        if self.optimizer not in {"adamw", "adam", "sgd"}:
            raise ValueError("optimizer must be one of: adamw, adam, sgd")
        if self.objective != "infonce":
            raise ValueError("objective must be 'infonce'")
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("config JSON must decode to an object")
        return cls.model_validate(payload)
