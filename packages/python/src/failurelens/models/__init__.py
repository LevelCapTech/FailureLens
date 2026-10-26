"""Model/backend registry subpackage."""

from __future__ import annotations

from failurelens.models.registry import (
    BASELINE_MODEL_ID,
    ModelInfo,
    get_model_info,
    list_models,
)

__all__ = ["BASELINE_MODEL_ID", "ModelInfo", "get_model_info", "list_models"]
