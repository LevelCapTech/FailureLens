"""Tests for the semantic dataset-record quality gates (`failurelens.quality`).

The committed fixture ``datasets/fixtures/sample_record.json`` is the canonical
conforming record and must pass every gate. Each failing case below starts from a
fresh copy of that fixture and mutates exactly the field under test, so a single
gate fires in isolation. Secret-looking strings are assembled at runtime so no
verbatim credential pattern is committed to source.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from failurelens.quality import (
    DEFAULT_REDISTRIBUTION_ALLOWLIST,
    GateRejectionReason,
    GateResult,
    check_pii,
    check_secret,
    validate_record,
    validate_record_set,
)

R = GateRejectionReason

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "datasets" / "fixtures" / "sample_record.json"


def load_fixture() -> dict[str, Any]:
    """Return a fresh, deep-copied dict of the committed sample record."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- happy path


def test_fixture_passes_all_gates() -> None:
    result = validate_record(load_fixture())
    assert result.passed
    assert result.rejection_reasons == ()


def test_fixture_passes_in_a_set_and_when_redistributed() -> None:
    # The fixture is Apache-2.0, which is on the default redistribution allowlist.
    [result] = validate_record_set([load_fixture()], redistributed=True)
    assert result.passed


# --------------------------------------------------------------------- single-gate table


def _set_license_unknown(record: dict[str, Any]) -> None:
    record["source_license"] = "unknown"


def _empty_candidates(record: dict[str, Any]) -> None:
    record["candidates"] = []


def _positive_only(record: dict[str, Any]) -> None:
    record["candidates"] = [c for c in record["candidates"] if c["label"] == "positive"]


def _negative_only(record: dict[str, Any]) -> None:
    record["candidates"] = [c for c in record["candidates"] if c["label"] == "negative"]


def _blank_rationale(record: dict[str, Any]) -> None:
    record["label_rationale"] = "   \n\t "


def _pre_not_passed(record: dict[str, Any]) -> None:
    record["pre_change_result"]["status"] = "failed"


def _post_not_failed(record: dict[str, Any]) -> None:
    record["post_change_result"]["status"] = "passed"


def _env_failure(record: dict[str, Any]) -> None:
    record["error_log"] = "ModuleNotFoundError: No module named 'requests'\n"


def _secret_aws(record: dict[str, Any]) -> None:
    record["error_log"] = "boto error, key " + "AKIA" + "IOSFODNN7EXAMPLE" + " rejected"


def _secret_password(record: dict[str, Any]) -> None:
    record["error_log"] = "connect failed: " + "pass" + "word=" + "n0trealvalue"


def _pii_email(record: dict[str, Any]) -> None:
    record["error_log"] = "notify owner alice@example.com of the failure"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_set_license_unknown, R.LICENSE_UNKNOWN),
        (_empty_candidates, R.EMPTY_CANDIDATES),
        (_positive_only, R.NO_NEGATIVE),
        (_negative_only, R.NO_POSITIVE),
        (_blank_rationale, R.EMPTY_RATIONALE),
        (_pre_not_passed, R.NOT_PASS_BEFORE_FAIL_AFTER),
        (_post_not_failed, R.NOT_PASS_BEFORE_FAIL_AFTER),
        (_env_failure, R.ENV_FAILURE),
        (_secret_aws, R.SECRET_DETECTED),
        (_secret_password, R.SECRET_DETECTED),
        (_pii_email, R.PII_DETECTED),
    ],
    ids=[
        "license-unknown",
        "empty-candidates",
        "no-negative",
        "no-positive",
        "empty-rationale",
        "pre-not-passed",
        "post-not-failed",
        "env-failure",
        "secret-aws",
        "secret-password",
        "pii-email",
    ],
)
def test_single_gate_rejections(
    mutate: Callable[[dict[str, Any]], None], expected: GateRejectionReason
) -> None:
    record = load_fixture()
    mutate(record)
    result = validate_record(record)
    assert not result.passed
    # Mutating one field fires exactly one gate.
    assert result.rejection_reasons == (expected,)


def test_short_rationale_is_rejected() -> None:
    record = load_fixture()
    record["label_rationale"] = "too short"
    assert validate_record(record).rejection_reasons == (R.EMPTY_RATIONALE,)


# ------------------------------------------------------------------------------- license


def test_license_not_redistributable_only_when_redistributed() -> None:
    record = load_fixture()
    record["source_license"] = "GPL-3.0-only"
    # Not flagged by default: redistribution is an explicit, human-governed choice.
    assert validate_record(record).passed
    # Flagged once the record is validated for redistribution.
    result = validate_record(record, redistributed=True)
    assert result.rejection_reasons == (R.LICENSE_NOT_REDISTRIBUTABLE,)


def test_license_unknown_takes_precedence_over_redistribution() -> None:
    record = load_fixture()
    record["source_license"] = "UNKNOWN"
    result = validate_record(record, redistributed=True)
    assert result.rejection_reasons == (R.LICENSE_UNKNOWN,)


def test_custom_allowlist_is_respected() -> None:
    record = load_fixture()
    record["source_license"] = "MPL-2.0"
    assert validate_record(record, allowlist={"MPL-2.0"}, redistributed=True).passed
    assert DEFAULT_REDISTRIBUTION_ALLOWLIST  # the default is non-empty


# ---------------------------------------------------------------------------- duplicates


def test_duplicate_record_id_in_set() -> None:
    results = validate_record_set([load_fixture(), load_fixture()])
    assert results[0].passed
    assert R.DUPLICATE in results[1].rejection_reasons


def test_near_duplicate_signature_detected_with_different_id() -> None:
    first = load_fixture()
    second = load_fixture()
    second["record_id"] = "a-different-id"  # same error_log + candidates signature
    results = validate_record_set([first, second])
    assert results[0].passed
    assert R.DUPLICATE in results[1].rejection_reasons


def test_distinct_records_are_not_duplicates() -> None:
    first = load_fixture()
    second = load_fixture()
    second["record_id"] = "second-record"
    second["error_log"] = "AssertionError: expected 3 but got 4\n"
    second["stack_trace"] = "app.py:9 -> pkg/mod.py:4 (check)"
    second["candidates"] = [
        {"location": {"file": "pkg/mod.py", "function": "check"}, "label": "positive"},
        {"location": {"file": "pkg/mod.py", "function": "run"}, "label": "negative"},
    ]
    results = validate_record_set([first, second])
    assert all(result.passed for result in results)


# --------------------------------------------------------------------------- accumulation


def test_multiple_violations_accumulate_in_gate_order() -> None:
    record = load_fixture()
    record["source_license"] = "unknown"
    record["candidates"] = []
    record["label_rationale"] = ""
    record["post_change_result"]["status"] = "passed"
    record["error_log"] = (
        "ImportError: boom; contact bob@example.org; " + "AKIA" + "IOSFODNN7EXAMPLE"
    )
    result = validate_record(record)
    assert not result.passed
    assert result.rejection_reasons == (
        R.LICENSE_UNKNOWN,
        R.EMPTY_CANDIDATES,
        R.EMPTY_RATIONALE,
        R.NOT_PASS_BEFORE_FAIL_AFTER,
        R.ENV_FAILURE,
        R.SECRET_DETECTED,
        R.PII_DETECTED,
    )


# -------------------------------------------------------------------------------- result


def test_to_dict_shape() -> None:
    passing = validate_record(load_fixture()).to_dict()
    assert passing == {"passed": True, "rejection_reasons": []}

    record = load_fixture()
    record["source_license"] = "unknown"
    failing = validate_record(record).to_dict()
    assert failing == {"passed": False, "rejection_reasons": ["license-unknown"]}


def test_gateresult_from_reasons_derives_passed() -> None:
    assert GateResult.from_reasons([]).passed
    result = GateResult.from_reasons([R.DUPLICATE])
    assert not result.passed
    assert result.rejection_reasons == (R.DUPLICATE,)


# ------------------------------------------------------------------------------ screeners


def test_check_secret_positive_and_negative() -> None:
    assert check_secret("token " + "AKIA" + "IOSFODNN7EXAMPLE")
    assert check_secret("auth " + "pass" + "word=" + "sup3rsecret")
    assert check_secret("header " + "ghp_" + "a" * 36)
    assert not check_secret("")
    assert not check_secret("ZeroDivisionError: division by zero")


def test_check_pii_positive_and_negative() -> None:
    assert check_pii("reach me at carol@example.com")
    assert check_pii("ssn 123-45-6789 on file")
    assert check_pii("call +1 (555) 123-4567 today")
    assert not check_pii("")
    assert not check_pii("app.py:21 -> core.py:12 (divide)")
