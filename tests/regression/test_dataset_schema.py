"""Regression guard: the dataset fixture conforms to the record schema.

This performs a lightweight structural check using only the standard library
(no ``jsonschema`` dependency): it verifies that the committed example record has
exactly the schema's top-level fields and that required nested fields are
present. Its purpose is to catch drift between ``record.schema.json`` and the
fixtures that document it.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "datasets" / "schemas" / "record.schema.json"
FIXTURE_PATH = REPO_ROOT / "datasets" / "fixtures" / "sample_record.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_schema_and_fixture_exist() -> None:
    assert SCHEMA_PATH.exists(), SCHEMA_PATH
    assert FIXTURE_PATH.exists(), FIXTURE_PATH


def test_fixture_has_all_required_top_level_fields() -> None:
    schema = _load(SCHEMA_PATH)
    fixture = _load(FIXTURE_PATH)
    for key in schema["required"]:
        assert key in fixture, f"fixture missing required field: {key}"


def test_fixture_has_no_unknown_top_level_fields() -> None:
    schema = _load(SCHEMA_PATH)
    fixture = _load(FIXTURE_PATH)
    allowed = set(schema["properties"])
    unknown = set(fixture) - allowed
    assert not unknown, f"fixture has fields not in schema: {sorted(unknown)}"


def test_fixture_candidates_are_well_formed() -> None:
    fixture = _load(FIXTURE_PATH)
    assert fixture["candidates"], "expected at least one candidate"
    labels = set()
    for candidate in fixture["candidates"]:
        assert "location" in candidate and "label" in candidate
        assert candidate["label"] in {"positive", "negative"}
        assert "file" in candidate["location"]
        labels.add(candidate["label"])
    # A useful record contrasts a cause with a non-cause.
    assert "positive" in labels


def test_fixture_execution_results_use_enum_status() -> None:
    schema = _load(SCHEMA_PATH)
    allowed = set(schema["$defs"]["execution_result"]["properties"]["status"]["enum"])
    fixture = _load(FIXTURE_PATH)
    assert fixture["pre_change_result"]["status"] in allowed
    assert fixture["post_change_result"]["status"] in allowed
    # The clean state must pass and the changed state must fail to be a valid case.
    assert fixture["pre_change_result"]["status"] == "passed"
    assert fixture["post_change_result"]["status"] == "failed"
