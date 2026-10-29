"""Dataset loading for the toy training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from failurelens.quality import validate_record
from training.config import DEFAULT_EXCLUDED_REPOS

EXCLUDED_REPOS = DEFAULT_EXCLUDED_REPOS
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_PATH = _REPO_ROOT / "datasets/schemas/record.schema.json"


@dataclass(frozen=True)
class TrainingExample:
    query: str
    positive: str
    negative: str
    source_repository: str
    dataset_version: str


def _validate_schema(record: dict[str, Any]) -> None:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(record), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "<root>"
        raise ValueError(
            "record failed JSON schema validation against "
            f"{_SCHEMA_PATH.as_posix()}: {location}: {first.message}"
        )


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
    return _candidate_text(pos_location), _candidate_text(neg_location)


def _candidate_text(location: dict[str, Any]) -> str:
    file_path = str(location.get("file", "")).strip()
    function = str(location.get("function") or "").strip()
    start_line = location.get("start_line")
    end_line = location.get("end_line")
    location_text = [file_path]
    if function:
        location_text.append(f"function={function}")
    if isinstance(start_line, int):
        if isinstance(end_line, int):
            location_text.append(f"lines={start_line}-{end_line}")
        else:
            location_text.append(f"line={start_line}")
    return " :: ".join(part for part in location_text if part)


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
    repository = str(record.get("source_repository", "")).strip()
    positive = f"{repository} :: {pos_file}".strip()
    negative = f"{repository} :: {neg_file}".strip()
    return [
        TrainingExample(
            query=query,
            positive=positive,
            negative=negative,
            source_repository=repository,
            dataset_version=dataset_version,
        )
    ]
