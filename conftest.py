"""Root conftest — shared pytest options for all test suites."""

from __future__ import annotations


def pytest_addoption(parser: object) -> None:
    parser.addoption(  # type: ignore[union-attr]
        "--run-ml-integration",
        action="store_true",
        default=False,
        help="Run ML integration tests that require the model cache and GPU.",
    )
