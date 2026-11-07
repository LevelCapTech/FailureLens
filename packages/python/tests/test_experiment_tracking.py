"""The default path stays offline and enabled runs have explicit lifecycles."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from failurelens import experiment_tracking as tracking
from failurelens.errors import MissingDependencyError


class RecordingClient:
    def __init__(self, tracking_uri: str) -> None:
        self.uri = tracking_uri
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, list[tuple[int, float]]] = {}
        self.artifacts: dict[str, Any] = {}
        self.tags: dict[str, str] = {}
        self.status = "RUNNING"
        self.experiment: Any = None
        self.fail_termination = False

    def get_experiment_by_name(self, name: str) -> Any:
        self.experiment_name = name
        return self.experiment

    def create_experiment(self, name: str, artifact_location: str | None) -> str:
        self.artifact_location = artifact_location
        return "experiment-1"

    def create_run(self, experiment_id: str, tags: dict[str, str], run_name: str | None) -> Any:
        self.tags = tags
        self.run_name = run_name
        return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    def log_batch(
        self,
        run_id: str,
        *,
        params: Sequence[Any] = (),
        metrics: Sequence[Any] = (),
        synchronous: bool,
    ) -> None:
        assert run_id == "run-1"
        assert synchronous
        self.params.update({param.key: param.value for param in params})
        for metric in metrics:
            self.metrics.setdefault(metric.key, []).append((metric.step, metric.value))

    def log_dict(self, run_id: str, data: dict[str, Any], filename: str) -> None:
        self.artifacts[filename] = data

    def set_terminated(self, run_id: str, status: str) -> None:
        if self.fail_termination:
            raise RuntimeError("tracking unavailable")
        self.status = status


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> RecordingClient:
    instance = RecordingClient("unused")

    def factory(tracking_uri: str) -> RecordingClient:
        instance.uri = tracking_uri
        return instance

    monkeypatch.setitem(sys.modules, "mlflow.tracking", SimpleNamespace(MlflowClient=factory))
    monkeypatch.setitem(
        sys.modules,
        "mlflow.entities",
        SimpleNamespace(
            Param=lambda key, value: SimpleNamespace(key=key, value=value),
            Metric=lambda key, value, timestamp, step: SimpleNamespace(
                key=key, value=value, step=step
            ),
        ),
    )
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    return instance


def test_disabled_never_imports_mlflow_or_creates_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_import(name: str) -> Any:
        pytest.fail(f"unexpected import: {name}")

    monkeypatch.setattr(tracking.importlib, "import_module", forbidden_import)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://unused.invalid")
    with tracking.track_experiment(None, kind="test", repository=tmp_path) as run:
        run.log_params({"seed": 7})
        run.log_metrics({"train.loss": 0.1})
        run.log_dict({}, "unused.json")
        assert run.run_id is None
    assert list(tmp_path.iterdir()) == []


def test_missing_dependency_is_actionable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> Any:
        raise ImportError("mlflow is absent")

    monkeypatch.setattr(tracking.importlib, "import_module", unavailable)
    with (
        pytest.raises(MissingDependencyError, match=r"failurelens\[tracking\]"),
        tracking.track_experiment(tracking.MLflowConfig(), kind="test", repository=tmp_path),
    ):
        pytest.fail("run should not start")


def test_local_run_metadata_and_step_history(client: RecordingClient, tmp_path: Path) -> None:
    with tracking.track_experiment(
        tracking.MLflowConfig(run_name="learning-curve"), kind="toy-training", repository=tmp_path
    ) as run:
        run.log_params({"seed": 7, "absent": None})
        run.log_metrics({"train.loss": 0.5}, step=1)
        run.log_metrics({"train.loss": 0.3}, step=2)
    assert client.uri == f"sqlite:///{tmp_path / '.mlflow/mlflow.db'}"
    assert client.artifact_location == (tmp_path / ".mlflow/artifacts").as_uri()
    assert client.run_name == "learning-curve"
    assert client.status == "FINISHED"
    assert client.params == {"seed": "7"}
    assert client.metrics["train.loss"] == [(1, 0.5), (2, 0.3)]
    assert client.metrics["run.duration_seconds"][0][1] >= 0
    assert client.tags["failurelens.kind"] == "toy-training"
    assert client.tags["mlflow.source.git.commit"] == "unknown"
    assert "dependencies" in client.artifacts["environment.json"]


def test_explicit_configuration_overrides_environment(
    client: RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://environment.invalid")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "environment")
    client.experiment = SimpleNamespace(experiment_id="existing")
    with tracking.track_experiment(
        tracking.MLflowConfig(tracking_uri="https://explicit.invalid", experiment_name="explicit"),
        kind="test",
        repository=tmp_path,
    ):
        pass
    assert client.uri == "https://explicit.invalid"
    assert client.experiment_name == "explicit"
    assert not (tmp_path / ".mlflow").exists()


@pytest.mark.parametrize("error", [ValueError("experiment failed"), KeyboardInterrupt()])
def test_failures_and_interruptions_end_run(
    client: RecordingClient, tmp_path: Path, error: BaseException
) -> None:
    with (
        pytest.raises(type(error)),
        tracking.track_experiment(tracking.MLflowConfig(), kind="test", repository=tmp_path),
    ):
        raise error
    assert client.status == "FAILED"


def test_tracking_outage_preserves_original_error(client: RecordingClient, tmp_path: Path) -> None:
    client.fail_termination = True
    with (
        pytest.raises(ValueError, match="original failure"),
        tracking.track_experiment(tracking.MLflowConfig(), kind="test", repository=tmp_path),
    ):
        raise ValueError("original failure")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurements_fail_instead_of_becoming_results(
    client: RecordingClient, tmp_path: Path, value: float
) -> None:
    with (
        pytest.raises(ValueError, match="non-finite"),
        tracking.track_experiment(tracking.MLflowConfig(), kind="test", repository=tmp_path) as run,
    ):
        run.log_metrics({"loss": value})
    assert client.status == "FAILED"


def test_cli_requires_opt_in_and_config_rejects_typos() -> None:
    parser = argparse.ArgumentParser()
    tracking.add_tracking_arguments(parser)
    assert tracking.tracking_from_args(parser, parser.parse_args([])) is None
    with pytest.raises(SystemExit):
        tracking.tracking_from_args(parser, parser.parse_args(["--mlflow-run-name", "unused"]))
    config = tracking.tracking_from_args(parser, parser.parse_args(["--mlflow"]))
    assert isinstance(config, tracking.MLflowConfig)
    with pytest.raises(ValueError):
        tracking.MLflowConfig.model_validate({"experment_name": "typo"})
