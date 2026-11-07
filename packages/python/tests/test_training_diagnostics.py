"""Overfitting indicators have documented, deterministic definitions."""

from __future__ import annotations

import pytest

from failurelens.experiment_tracking import MLflowConfig, training_diagnostics


def test_healthy_curve_has_no_signal() -> None:
    d = training_diagnostics([0.8, 0.6, 0.5], [0.9, 0.7, 0.6])
    assert d["diag.best_epoch"] == 3
    assert d["diag.epochs_since_best"] == 0
    assert d["diag.overfit_signal"] == 0.0
    assert d["diag.final_gap"] == pytest.approx(0.1)
    assert d["diag.gap_trend"] == pytest.approx(0.0, abs=1e-12)


def test_validation_worsening_while_training_improves_is_flagged() -> None:
    d = training_diagnostics([0.8, 0.5, 0.3], [0.7, 0.6, 0.9])
    assert d["diag.best_epoch"] == 2
    assert d["diag.epochs_since_best"] == 1
    assert d["diag.overfit_signal"] == 1.0
    assert d["diag.gap_trend"] > 0


def test_noise_in_both_curves_is_not_flagged() -> None:
    assert training_diagnostics([0.4, 0.5], [0.7, 0.8])["diag.overfit_signal"] == 0.0


def test_single_epoch_and_validation() -> None:
    assert training_diagnostics([0.5], [0.6])["diag.gap_trend"] == 0.0
    for bad in (([], []), ([1.0], [1.0, 2.0]), ([float("nan")], [1.0])):
        with pytest.raises(ValueError):
            training_diagnostics(*bad)


@pytest.mark.parametrize("version", ["v0.1.0", "v0.2.0-candidate-1", "v10.0.3-rc.1"])
def test_valid_model_versions(version: str) -> None:
    assert MLflowConfig(model_version=version).model_version == version


@pytest.mark.parametrize("version", ["0.1.0", "v1", "v1.0.0 ", "v1.0.0;rm", "latest"])
def test_invalid_model_versions(version: str) -> None:
    with pytest.raises(ValueError):
        MLflowConfig(model_version=version)
