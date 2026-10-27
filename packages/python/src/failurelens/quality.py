"""Semantic quality gates for dataset records (beyond JSON Schema).

``datasets/schemas/record.schema.json`` is a necessary-but-not-sufficient guard:
it accepts ``source_license: "unknown"`` and an *empty* ``candidates`` array and
does not require a non-empty ``label_rationale``. The gates here close exactly
those gaps, enforcing the "Semantic quality gates (beyond JSON Schema)" table in
``docs/data-contract.md`` and recording a **rejection reason** for every dropped
record.

The gates are **pure** and **dependency-free**: they read a single record
mapping (or a sequence of them), never touch the network or filesystem, and never
mutate their input. Scores are not invented and no dataset content is fabricated;
this module only *validates* records that a caller already holds.

The secret/PII checks (:func:`check_secret`, :func:`check_pii`) are deterministic,
regex/entropy-based **screeners** -- they reduce risk but are not guarantees; a
clean result does not prove a record is free of secrets or personal data.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "DEFAULT_REDISTRIBUTION_ALLOWLIST",
    "GateRejectionReason",
    "GateResult",
    "check_pii",
    "check_secret",
    "validate_record",
    "validate_record_set",
]


class GateRejectionReason(StrEnum):
    """Why a record was rejected by a semantic gate.

    Each member's value is the kebab-case reason string used in the
    ``docs/data-contract.md`` gate table and recorded on dropped records.
    """

    LICENSE_UNKNOWN = "license-unknown"
    LICENSE_NOT_REDISTRIBUTABLE = "license-not-redistributable"
    NO_POSITIVE = "no-positive"
    NO_NEGATIVE = "no-negative"
    EMPTY_CANDIDATES = "empty-candidates"
    EMPTY_RATIONALE = "empty-rationale"
    NOT_PASS_BEFORE_FAIL_AFTER = "not-pass-before-fail-after"
    ENV_FAILURE = "env-failure"
    SECRET_DETECTED = "secret-detected"
    PII_DETECTED = "pii-detected"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class GateResult:
    """Outcome of running the semantic gates on one record.

    ``passed`` is ``True`` only when ``rejection_reasons`` is empty. A record may
    violate several gates at once; all reasons are reported, in gate order.
    """

    passed: bool
    rejection_reasons: tuple[GateRejectionReason, ...] = ()

    @classmethod
    def from_reasons(cls, reasons: Sequence[GateRejectionReason]) -> GateResult:
        """Build a result from accumulated reasons, deriving ``passed``."""
        ordered = tuple(reasons)
        return cls(passed=not ordered, rejection_reasons=ordered)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping for reporting."""
        return {
            "passed": self.passed,
            "rejection_reasons": [reason.value for reason in self.rejection_reasons],
        }


#: Permissive SPDX ids allowed for *redistributed* records by default. The list is
#: intentionally small and human-governed; callers pass their own allowlist when a
#: different redistribution policy applies. "Verified" remains a human decision --
#: being on this list only clears the redistribution check, not rights review.
DEFAULT_REDISTRIBUTION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "CC0-1.0",
    }
)

#: Minimum trimmed length for ``label_rationale``. Presence plus a non-trivial
#: length is the gate; this module does not NLP-score whether the text truly
#: explains the root cause (that stays a human-review concern).
_MIN_RATIONALE_LENGTH = 20

#: Conservative, documented keyword signatures of install/network/environment
#: failures that are not caused by the change under test. Matched case-sensitively
#: as substrings of ``error_log``. This is a heuristic screener, not a parser.
_ENV_FAILURE_SIGNATURES: tuple[str, ...] = (
    "ModuleNotFoundError",
    "ImportError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "TimeoutError",
    "socket.timeout",
    "No such file or directory",
    "Temporary failure in name resolution",
    "Name or service not known",
    "Could not find a version that satisfies the requirement",
    "No matching distribution found",
    "Failed building wheel",
    "error: subprocess-exited-with-error",
    "Max retries exceeded",
    "urlopen error",
    "Read timed out",
    "Connection refused",
)

# Secret screeners. Explicit, well-known credential shapes plus a conservative
# high-entropy fallback. These are screeners, not guarantees.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),  # AWS temporary access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),  # GitHub tokens
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),  # OpenAI-style secret key
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),  # Slack token
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),  # Google API key
    re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|token)\b"
        r"\s*[:=]\s*[^\s'\"]{3,}"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}\b"),
)

#: A token that could plausibly be a secret, used only by the entropy fallback.
_SECRET_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_\-]{32,}")
_SECRET_ENTROPY_BITS = 4.5

# PII screeners.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
#: A loose phone-like run; confirmed only when it has >=10 digits and a separator,
#: so plain short line numbers in a traceback are not mistaken for phone numbers.
_PHONE_CANDIDATE_RE = re.compile(r"\+?\d[\d\s().\-]{7,}\d")


def _shannon_entropy(value: str) -> float:
    """Return the Shannon entropy (bits/char) of ``value``."""
    if not value:
        return 0.0
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in Counter(value).values())


def check_secret(text: str) -> bool:
    """Return ``True`` if ``text`` looks like it contains a secret/credential.

    Deterministic and dependency-free. Detects well-known credential shapes
    (private-key blocks, ``AKIA``/``ASIA`` AWS keys, ``sk-``/``gh*_``/``xox*``/
    ``AIza`` tokens, ``Bearer`` tokens, and labelled credential assignments such
    as a password or secret set with ``=`` or ``:``) plus a conservative
    high-entropy fallback. This is a *screener*: a ``False`` result is not a
    guarantee that no secret is present.
    """
    if not text:
        return False
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return True
    return any(
        _shannon_entropy(token) >= _SECRET_ENTROPY_BITS for token in _SECRET_TOKEN_RE.findall(text)
    )


def check_pii(text: str) -> bool:
    """Return ``True`` if ``text`` looks like it contains personal data.

    Deterministic and dependency-free. Detects email addresses, US SSN-like
    ``NNN-NN-NNNN`` runs, and phone-number-like sequences (>=10 digits with a
    separator). This is a *screener*, not a guarantee.
    """
    if not text:
        return False
    if _EMAIL_RE.search(text) or _SSN_RE.search(text):
        return True
    for match in _PHONE_CANDIDATE_RE.finditer(text):
        token = match.group()
        digits = sum(char.isdigit() for char in token)
        has_separator = token.startswith("+") or any(char in " ().-" for char in token)
        if 10 <= digits <= 15 and has_separator:
            return True
    return False


def _candidate_labels(record: Mapping[str, Any]) -> set[str]:
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes):
        return set()
    return {str(item.get("label")) for item in candidates if isinstance(item, Mapping)}


def _result_status(record: Mapping[str, Any], key: str) -> str | None:
    section = record.get(key)
    if isinstance(section, Mapping):
        status = section.get("status")
        return status if isinstance(status, str) else None
    return None


def _screen_text(record: Mapping[str, Any]) -> str:
    parts = [
        value for key in ("error_log", "stack_trace") if isinstance((value := record.get(key)), str)
    ]
    return "\n".join(parts)


def _check_license(
    record: Mapping[str, Any], allowlist: Collection[str], redistributed: bool
) -> list[GateRejectionReason]:
    raw = record.get("source_license")
    value = str(raw).strip() if raw is not None else ""
    if value == "" or value.casefold() == "unknown":
        return [GateRejectionReason.LICENSE_UNKNOWN]
    if redistributed and value not in allowlist:
        return [GateRejectionReason.LICENSE_NOT_REDISTRIBUTABLE]
    return []


def _check_candidates(record: Mapping[str, Any]) -> list[GateRejectionReason]:
    candidates = record.get("candidates")
    if not candidates:
        return [GateRejectionReason.EMPTY_CANDIDATES]
    labels = _candidate_labels(record)
    reasons: list[GateRejectionReason] = []
    if "positive" not in labels:
        reasons.append(GateRejectionReason.NO_POSITIVE)
    if "negative" not in labels:
        reasons.append(GateRejectionReason.NO_NEGATIVE)
    return reasons


def _check_rationale(record: Mapping[str, Any]) -> list[GateRejectionReason]:
    raw = record.get("label_rationale")
    text = str(raw).strip() if raw is not None else ""
    if len(text) < _MIN_RATIONALE_LENGTH:
        return [GateRejectionReason.EMPTY_RATIONALE]
    return []


def _check_validity(record: Mapping[str, Any]) -> list[GateRejectionReason]:
    pre = _result_status(record, "pre_change_result")
    post = _result_status(record, "post_change_result")
    if pre != "passed" or post != "failed":
        return [GateRejectionReason.NOT_PASS_BEFORE_FAIL_AFTER]
    return []


def _check_environment(record: Mapping[str, Any]) -> list[GateRejectionReason]:
    error_log = record.get("error_log")
    if isinstance(error_log, str) and any(
        signature in error_log for signature in _ENV_FAILURE_SIGNATURES
    ):
        return [GateRejectionReason.ENV_FAILURE]
    return []


def _check_secrets_and_pii(record: Mapping[str, Any]) -> list[GateRejectionReason]:
    text = _screen_text(record)
    reasons: list[GateRejectionReason] = []
    if check_secret(text):
        reasons.append(GateRejectionReason.SECRET_DETECTED)
    if check_pii(text):
        reasons.append(GateRejectionReason.PII_DETECTED)
    return reasons


def validate_record(
    record: Mapping[str, Any],
    *,
    allowlist: Collection[str] = DEFAULT_REDISTRIBUTION_ALLOWLIST,
    redistributed: bool = False,
) -> GateResult:
    """Run every per-record semantic gate and return all failures.

    Gates are applied in the ``docs/data-contract.md`` order -- license,
    candidates, rationale, validity, environment, secrets/PII -- and **all**
    violations are accumulated (evaluation does not stop at the first failure).
    The duplicate gate is cross-record and lives in :func:`validate_record_set`.

    Args:
        record: A single dataset record mapping (not mutated).
        allowlist: SPDX ids permitted for redistributed records. Redistribution
            is human-governed, so the allowlist is an explicit parameter.
        redistributed: When ``True``, a known license outside ``allowlist`` is
            rejected as ``license-not-redistributable``. When ``False`` (default)
            only the ``unknown`` sentinel is rejected.
    """
    reasons: list[GateRejectionReason] = []
    reasons += _check_license(record, allowlist, redistributed)
    reasons += _check_candidates(record)
    reasons += _check_rationale(record)
    reasons += _check_validity(record)
    reasons += _check_environment(record)
    reasons += _check_secrets_and_pii(record)
    return GateResult.from_reasons(reasons)


def _record_signature(record: Mapping[str, Any]) -> str:
    """Return a stable near-duplicate signature for a record.

    The coarse signal is the whitespace-normalized stack trace (falling back to
    ``error_log``) plus the sorted ``(file, function)`` locations of the
    candidates. Records that differ only in formatting collide; this is a
    deliberately low-resolution near-duplicate signal, not a semantic diff.
    """
    trace = record.get("stack_trace") or record.get("error_log") or ""
    normalized_trace = " ".join(str(trace).split())
    locations: list[tuple[str, str]] = []
    candidates = record.get("candidates")
    if isinstance(candidates, Sequence) and not isinstance(candidates, str | bytes):
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            location = item.get("location")
            if isinstance(location, Mapping):
                locations.append((str(location.get("file")), str(location.get("function"))))
    payload = repr((normalized_trace, sorted(locations)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_record_set(
    records: Sequence[Mapping[str, Any]],
    *,
    allowlist: Collection[str] = DEFAULT_REDISTRIBUTION_ALLOWLIST,
    redistributed: bool = False,
) -> list[GateResult]:
    """Validate every record *and* apply the cross-record duplicate gate.

    Each record is checked with :func:`validate_record`; in addition, a record is
    rejected as ``duplicate`` when an earlier record in the sequence shares its
    ``record_id`` (the hard gate) or its near-duplicate signature (see
    :func:`_record_signature`). The first occurrence is retained; later
    collisions are flagged.
    """
    results: list[GateResult] = []
    seen_ids: set[str] = set()
    seen_signatures: set[str] = set()
    for record in records:
        base = validate_record(record, allowlist=allowlist, redistributed=redistributed)
        reasons = list(base.rejection_reasons)
        record_id = record.get("record_id")
        signature = _record_signature(record)
        is_duplicate = (
            record_id is not None and str(record_id) in seen_ids
        ) or signature in seen_signatures
        if is_duplicate:
            reasons.append(GateRejectionReason.DUPLICATE)
        if record_id is not None:
            seen_ids.add(str(record_id))
        seen_signatures.add(signature)
        results.append(GateResult.from_reasons(reasons))
    return results
