"""Dataset loading for the toy training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from failurelens.quality import validate_record
from training.config import DEFAULT_EXCLUDED_REPOS

EXCLUDED_REPOS = DEFAULT_EXCLUDED_REPOS


@dataclass(frozen=True)
class TrainingExample:
    query: str
    positive: str
    negative: str
    source_repository: str
    dataset_version: str


def _validate_schema(record: dict[str, Any]) -> None:
    required = {
        "record_id",
        "dataset_id",
        "source_repository",
        "source_commit",
        "source_license",
        "language",
        "test_command",
        "pre_change_result",
        "post_change_result",
        "error_log",
        "candidates",
        "label_rationale",
        "generated_at",
        "generator_version",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"record missing required fields: {', '.join(missing)}")
    if not isinstance(record["error_log"], str):
        raise ValueError("error_log must be a string")
    if not isinstance(record["candidates"], list):
        raise ValueError("candidates must be a list")
    if not isinstance(record["label_rationale"], str):
        raise ValueError("label_rationale must be a string")


def _validate_repository(record: dict[str, Any]) -> str:
    repository = str(record.get("source_repository", "")).strip()
    if repository in EXCLUDED_REPOS:
        raise ValueError(f"dataset source repository is excluded from training: {repository}")
    if not repository:
        raise ValueError("source_repository must not be empty")
    return repository


def _positive_negative_examples(record: dict[str, Any]) -> tuple[str, str]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    positive = next((item for item in candidates if item.get("label") == "positive"), None)
    negative = next((item for item in candidates if item.get("label") == "negative"), None)
    if positive is None or negative is None:
        raise ValueError("record must contain both a positive and a negative candidate")
    if not isinstance(positive, dict) or not isinstance(negative, dict):
        raise ValueError("candidate entries must be objects")
    pos_location = positive.get("location")
    neg_location = negative.get("location")
    if not isinstance(pos_location, dict) or not isinstance(neg_location, dict):
        raise ValueError("candidate locations must be objects")
    return str(pos_location.get("file", "positive")), str(neg_location.get("file", "negative"))


def load_training_examples(dataset_path: str | Path | None = None) -> list[TrainingExample]:
    """Load an approved local fixture into (query, positive, negative) training pairs."""
    path = (
        Path(dataset_path)
        if dataset_path is not None
        else Path("datasets/fixtures/sample_record.json")
    )
    if not path.exists():
        raise ValueError(f"dataset file does not exist: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("dataset record must be a JSON object")

    _validate_schema(record)
    repository = _validate_repository(record)
    result = validate_record(record)
    if not result.passed:
        raise ValueError(f"record failed semantic quality gates: {result.rejection_reasons}")

    pos_file, neg_file = _positive_negative_examples(record)
    dataset_version = str(record.get("dataset_id", "fixture-v1"))
    query = str(record.get("error_log", "")).strip()
    positive = (
        f"{record.get('label_rationale', '')} :: {pos_file} :: {record.get('source_repository', '')}"
    ).strip()
    negative = (
        f"{record.get('label_rationale', '')} :: {neg_file} :: {record.get('source_repository', '')}"
    ).strip()
    return [
        TrainingExample(
            query=query,
            positive=positive,
            negative=negative,
            source_repository=repository,
            dataset_version=dataset_version,
        )
    ]
