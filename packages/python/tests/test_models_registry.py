"""Tests for the honest model/backend registry."""

from __future__ import annotations

from failurelens.models.registry import (
    BASELINE_MODEL_ID,
    get_model_info,
    list_models,
)


def test_baseline_is_available_and_released() -> None:
    info = get_model_info(BASELINE_MODEL_ID)
    assert info is not None
    assert info.available is True
    assert info.released is True
    assert not info.requires_extras


def test_unknown_model_returns_none() -> None:
    assert get_model_info("does-not-exist") is None


def test_list_models_includes_baseline() -> None:
    ids = {info.id for info in list_models()}
    assert BASELINE_MODEL_ID in ids


def test_external_models_are_not_claimed_as_bundled() -> None:
    # Any non-baseline model must be marked as needing extras (i.e. not bundled)
    # so the CLI/API never implies FailureLens ships trained weights.
    for info in list_models():
        if info.id == BASELINE_MODEL_ID:
            continue
        assert info.available is False
        assert info.requires_extras
