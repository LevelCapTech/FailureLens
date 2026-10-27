"""Regression guard: the dataset fixture passes the semantic quality gates.

The sibling ``test_dataset_schema.py`` checks the fixture *structurally* against
``record.schema.json``. This module adds the complementary *semantic* check: the
committed canonical record must also satisfy every gate implemented in
``failurelens.quality`` (license, candidates, rationale, validity, environment,
secrets/PII, duplicates), so drift that a schema cannot catch is caught here.
"""

from __future__ import annotations

import json
from pathlib import Path

from failurelens.quality import validate_record, validate_record_set

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "datasets" / "fixtures" / "sample_record.json"


def _load() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_passes_every_semantic_gate() -> None:
    result = validate_record(_load())
    assert result.passed, result.rejection_reasons
    assert result.rejection_reasons == ()


def test_fixture_passes_when_validated_for_redistribution() -> None:
    # The fixture's Apache-2.0 license is on the default redistribution allowlist.
    [result] = validate_record_set([_load()], redistributed=True)
    assert result.passed, result.rejection_reasons
