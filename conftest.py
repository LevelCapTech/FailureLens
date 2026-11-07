"""Root conftest — shared pytest options for all test suites."""

from __future__ import annotations


def pytest_addoption(parser: object) -> None:
    parser.addoption(  # type: ignore[union-attr]
        "--run-ml-integration",
        action="store_true",
        default=False,
        help="Run ML integration tests that require the model cache and GPU.",
    )


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_notifications(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Never write to the user's notification log or contact real channels in tests."""
    monkeypatch.setenv("FAILURELENS_NOTIFY", "0")
    monkeypatch.setenv(
        "FAILURELENS_NOTIFY_CONFIG", str(tmp_path_factory.getbasetemp() / "no-notify-config.json")
    )
