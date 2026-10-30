"""Opt-in, local-only held-out input and candidate-parity boundary.

No private path or record content is included in raised errors. This module never
runs a ranker or emits a report as a side effect of loading inputs.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from jsonschema import Draft202012Validator

from failurelens.quality import validate_record_set

_SCHEMA = Path(__file__).resolve().parents[1] / "datasets/schemas/record.schema.json"


class PrivateInputError(ValueError):
    """Redacted validation or parity failure."""


@dataclass(frozen=True)
class Candidate:
    id: str
    file_id: str
    function_id: str | None
    source_text: str
    source_sha256: str


@dataclass(frozen=True)
class RankRequest:
    query: str
    candidates: tuple[Candidate, ...]
    top_k: int
    digest: str


@dataclass(frozen=True)
class RankResult:
    ordered_ids: tuple[str, ...]
    request_digest: str
    top_k: int


class Ranker(Protocol):
    def __call__(self, request: RankRequest) -> RankResult: ...


@dataclass(frozen=True)
class PrivateCase:
    """The labels are held here, never in RankRequest or Candidate."""

    request: RankRequest
    positive_ids: frozenset[str]
    positive_file_ids: frozenset[str]
    positive_function_ids: frozenset[str] | None


def _reject(reason: str) -> None:
    raise PrivateInputError(f"private evaluation rejected: {reason}") from None


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _reject("unreadable input")


def _git_source(root: Path, commit: str, file: str) -> str:
    # Git object reads give the exact pinned revision, regardless of worktree state.
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{file}"],
            capture_output=True,
            check=True,
            timeout=15,
        )
        return result.stdout.decode("utf-8")
    except (OSError, subprocess.SubprocessError, UnicodeError):
        _reject("missing pinned source")


def _location_id(location: dict[str, Any]) -> str:
    file = location["file"]
    function = location.get("function")
    start = location.get("start_line")
    end = location.get("end_line")
    return f"{file}::{function or '<file>'}:{start or 0}-{end or 0}"


def _candidate(record_candidate: dict[str, Any], root: Path, commit: str) -> Candidate:
    location = record_candidate["location"]
    file = location["file"]
    pure = PurePosixPath(file)
    if not file or pure.is_absolute() or ".." in pure.parts or pure.suffix != ".py":
        _reject("invalid source location")
    source = _git_source(root, commit, file)
    start = location.get("start_line")
    end = location.get("end_line")
    if (start is None) != (end is None) or (start is not None and (end < start)):
        _reject("invalid source location")
    if start is not None:
        lines = source.splitlines(keepends=True)
        if start < 1 or end > len(lines):
            _reject("missing source location")
        source = "".join(lines[start - 1 : end])
    if not source.strip():
        _reject("missing source location")
    function = location.get("function")
    if function is not None and (not function.strip() or start is None):
        _reject("missing function source location")
    return Candidate(
        id=_location_id(location),
        file_id=file,
        function_id=f"{file}::{function}" if function else None,
        source_text=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )


def _request_digest(query: str, candidates: tuple[Candidate, ...], top_k: int) -> str:
    return _digest(
        {
            "query": query,
            "top_k": top_k,
            "candidates": [candidate.__dict__ for candidate in candidates],
        }
    )


def _load_private_cases(
    records_path: Path,
    expected_sha256: str,
    source_manifest_path: Path,
    train_manifest_path: Path,
    *,
    top_k: int,
    production: bool = True,
) -> tuple[PrivateCase, ...]:
    """Load runtime-only records and approved manifest; fail closed without data in errors.

    Source manifest: {"repositories": {repo_id: {"path": local_git_path,
    "commit": sha40, "rights_cleared": true}}}. Train manifest:
    {"approval_id": nonempty_string, "training_repositories": [repo_id, ...]}.
    All four inputs are caller-supplied. No repository paths are embedded here.
    """
    if type(top_k) is not int or top_k < 1:
        _reject("invalid top_k")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        _reject("invalid expected digest")
    try:
        raw = records_path.read_bytes()
    except OSError:
        _reject("unreadable input")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        _reject("digest mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        _reject("invalid record set")
    records = payload.get("records") if isinstance(payload, dict) else payload
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(r, dict) for r in records)
    ):
        _reject("invalid record set")
    schema = _read_json(_SCHEMA)
    validator = Draft202012Validator(schema)
    if any(not validator.is_valid(record) for record in records):
        _reject("schema violation")
    if any(not result.passed for result in validate_record_set(records, redistributed=True)):
        _reject("quality violation")
    sources = _read_json(source_manifest_path)
    train = _read_json(train_manifest_path)
    if not isinstance(sources, dict) or not isinstance(sources.get("repositories"), dict):
        _reject("invalid source manifest")
    if (
        not isinstance(train, dict)
        or not isinstance(train.get("approval_id"), str)
        or not train["approval_id"].strip()
        or not isinstance(train.get("training_repositories"), list)
        or any(not isinstance(x, str) or not x.strip() for x in train["training_repositories"])
    ):
        _reject("invalid train manifest")
    repos = {record["source_repository"] for record in records}
    if any(not repo.strip() for repo in repos):
        _reject("missing repository")
    if production and (len(records) < 30 or len(repos) < 3):
        _reject("insufficient evaluation set")
    if repos & set(train["training_repositories"]):
        _reject("train/eval repository overlap")
    source_map = sources["repositories"]
    for repo in repos:
        entry = source_map.get(repo)
        if not isinstance(entry, dict) or entry.get("rights_cleared") is not True:
            _reject("missing rights-cleared source pin")
        commit = entry.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
        ):
            _reject("invalid source pin")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            _reject("missing source path")
    cases = []
    for record in records:
        entry = source_map[record["source_repository"]]
        if record["source_commit"] != entry["commit"]:
            _reject("source pin mismatch")
        candidates = tuple(
            _candidate(candidate, Path(entry["path"]), entry["commit"])
            for candidate in record["candidates"]
        )
        ids = [candidate.id for candidate in candidates]
        if len(ids) != len(set(ids)):
            _reject("duplicate candidate ID")
        positives = tuple(
            candidate
            for candidate, raw_candidate in zip(candidates, record["candidates"], strict=True)
            if raw_candidate["label"] == "positive"
        )
        if not positives:
            _reject("missing positive source location")
        query = record["error_log"]
        request = RankRequest(query, candidates, top_k, _request_digest(query, candidates, top_k))
        cases.append(
            PrivateCase(
                request,
                frozenset(candidate.id for candidate in positives),
                frozenset(candidate.file_id for candidate in positives),
                (
                    frozenset(
                        candidate.function_id for candidate in positives if candidate.function_id
                    )
                    if all(candidate.function_id for candidate in positives)
                    else None
                ),
            )
        )
    return tuple(cases)


def load_private_cases(
    records_path: Path,
    expected_sha256: str,
    source_manifest_path: Path,
    train_manifest_path: Path,
    *,
    top_k: int,
    production: bool = True,
) -> tuple[PrivateCase, ...]:
    """Load caller-supplied private inputs, redacting all unexpected failures."""
    try:
        return _load_private_cases(
            records_path,
            expected_sha256,
            source_manifest_path,
            train_manifest_path,
            top_k=top_k,
            production=production,
        )
    except PrivateInputError:
        raise
    except Exception:
        _reject("invalid private input")


def compare_rankers(
    case: PrivateCase, baseline: Ranker, model: Ranker
) -> tuple[RankResult, RankResult]:
    """Run both on the same immutable, unlabeled request; require complete permutations."""
    request = case.request
    original = _request_digest(request.query, request.candidates, request.top_k)
    if original != request.digest:
        _reject("changed canonical request")
    results = []
    for ranker in (baseline, model):
        try:
            result = ranker(request)
        except Exception:
            _reject("ranker failed")
        if _request_digest(request.query, request.candidates, request.top_k) != original:
            _reject("changed canonical request")
        if (
            not isinstance(result, RankResult)
            or result.request_digest != original
            or type(result.top_k) is not int
            or result.top_k != request.top_k
        ):
            _reject("ranker parity violation")
        ordered = result.ordered_ids
        expected = {candidate.id for candidate in request.candidates}
        if (
            not isinstance(ordered, tuple)
            or any(not isinstance(item, str) for item in ordered)
            or len(ordered) != len(expected)
            or set(ordered) != expected
        ):
            _reject("ranker candidate mismatch")
        results.append(result)
    return results[0], results[1]
