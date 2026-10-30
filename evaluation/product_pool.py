"""Evaluator-only shipped-baseline retrieval on a verified complete failing tree.

The runtime manifests, trees, mutation text, and records remain local. This
module emits no case data or reports and never invokes a model on import.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from evaluation.private_adapter import (
    Candidate,
    PrivateCase,
    PrivateInputError,
    RankRequest,
    RankResult,
    _location_id,
    _query,
    _request_digest,
    load_private_cases,
)

from failurelens import FailureLens
from failurelens.config import IndexConfig
from failurelens.indexing.repository import _is_excluded, iter_source_files
from failurelens.schemas import AnalysisStatus


def _reject(reason: str) -> None:
    raise PrivateInputError(f"private evaluation rejected: {reason}") from None


def _hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def _path(value: object) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        _reject("invalid tree pin")
    return Path(value)


def _relative(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or "\\" in value
    ):
        _reject("invalid tree path")
    return value


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, check=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        _reject("unavailable source commit")


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _tree_blobs(repo: Path, commit: str, config: IndexConfig) -> dict[str, str]:
    if _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").strip().decode() != commit:
        _reject("source commit mismatch")
    output = _git(repo, "ls-tree", "-rz", "--full-tree", commit)
    blobs: dict[str, str] = {}
    for entry in output.split(b"\0"):
        if not entry:
            continue
        try:
            mode_type_oid, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = mode_type_oid.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeError):
            _reject("invalid source tree")
        if (
            not path.endswith(config.include_suffixes)
            or any(part in config.exclude_dirs for part in PurePosixPath(path).parts[:-1])
            or _is_excluded(path, config)
        ):
            continue
        if mode not in {"100644", "100755"} or kind != "blob" or not _hex(oid, 40):
            _reject("unsupported indexed source")
        blobs[path] = oid
    return blobs


def verify_failing_tree(entry: dict[str, Any], commit: str) -> Path:
    """Require every product-indexable file to match pinned Git plus one exact edit."""
    if not isinstance(entry, dict) or set(entry) != {
        "base_repository",
        "root",
        "commit",
        "indexed_files",
        "mutation",
    }:
        _reject("invalid tree pin")
    if entry["commit"] != commit or not _hex(commit, 40):
        _reject("source pin mismatch")
    base, root = _path(entry["base_repository"]), _path(entry["root"])
    if not base.is_dir() or not root.is_dir() or base.is_symlink() or root.is_symlink():
        _reject("unavailable failing tree")
    pins, mutation = entry["indexed_files"], entry["mutation"]
    if (
        not isinstance(pins, dict)
        or not isinstance(mutation, dict)
        or set(mutation) != {"file", "function", "old", "new"}
    ):
        _reject("invalid tree pin")
    changed = _relative(mutation["file"])
    old, new, function = mutation["old"], mutation["new"], mutation["function"]
    if (
        not isinstance(old, str)
        or not old
        or not isinstance(new, str)
        or old == new
        or not isinstance(function, str)
        or not function
    ):
        _reject("invalid mutation recipe")
    config = IndexConfig()
    blobs = _tree_blobs(base, commit, config)
    if not blobs or changed not in blobs or set(pins) != set(blobs):
        _reject("partial failing tree")
    actual = {name for _, name in iter_source_files(root, config)}
    if actual != set(blobs):
        _reject("partial failing tree")
    for name, oid in blobs.items():
        file = root / name
        if file.is_symlink() or not file.is_file() or not _hex(pins[name], 64):
            _reject("invalid indexed file pin")
        try:
            raw = file.read_bytes()
        except OSError:
            _reject("unreadable indexed source")
        if config.max_file_bytes and len(raw) > config.max_file_bytes:
            _reject("indexed source exceeds product limit")
        if hashlib.sha256(raw).hexdigest() != pins[name]:
            _reject("indexed file digest mismatch")
        if name == changed:
            original = _git(base, "cat-file", "blob", oid)
            try:
                text = original.decode("utf-8")
                matches = [
                    node
                    for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == function
                ]
            except (UnicodeError, SyntaxError):
                _reject("mutation provenance mismatch")
            if not matches or matches[-1].end_lineno is None:
                _reject("mutation provenance mismatch")
            selected = matches[-1]
            lines = text.splitlines(keepends=True)
            body = "".join(lines[selected.lineno - 1 : selected.end_lineno])
            if body.count(old) != 1:
                _reject("mutation provenance mismatch")
            lines[selected.lineno - 1 : selected.end_lineno] = [body.replace(old, new, 1)]
            if "".join(lines).encode() != raw:
                _reject("mutation provenance mismatch")
        elif _git_blob(raw) != oid:
            _reject("failing tree differs from source pin")
    return root


def _product_candidate(output: Any, root: Path, index: Any) -> Candidate:
    file = _relative(output.file)
    indexed = index.file(file)
    if indexed is None:
        _reject("missing retrieved source")
    try:
        source = (root / file).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        _reject("unreadable retrieved source")
    function, start, end = output.function, output.start_line, output.end_line
    if function is None:
        if start is not None or end is not None:
            _reject("ambiguous retrieved source")
        span = source
    else:
        if start is None or end is None or start < 1 or end < start:
            _reject("missing function span")
        matches = [
            unit
            for unit in indexed.units
            if unit.name == function and unit.start_line == start and unit.end_line == end
        ]
        same_name = [unit for unit in indexed.units if unit.name == function]
        if len(matches) != 1 or len(same_name) != 1:
            _reject("ambiguous or missing function span")
        lines = source.splitlines(keepends=True)
        if end > len(lines):
            _reject("missing function span")
        span = "".join(lines[start - 1 : end])
    if not span.strip():
        _reject("empty retrieved source")
    location = {"file": file, "function": function, "start_line": start, "end_line": end}
    return Candidate(
        _location_id(location),
        file,
        f"{file}::{function}" if function else None,
        span,
        hashlib.sha256(span.encode()).hexdigest(),
    )


class ProductBaseline:
    """Frozen ordered product output keyed by the exact unlabeled request digest.

    Timing this adapter measures lookup only; product retrieval is precomputed.
    """

    latency_scope = "precomputed-product-order-lookup"

    def __init__(self, orders: dict[str, tuple[str, ...]]) -> None:
        self._orders = dict(orders)

    def __call__(self, request: RankRequest) -> RankResult:
        order = self._orders.get(request.digest)
        if order is None:
            _reject("unrecognized product retrieval")
        return RankResult(order, request.digest, request.top_k)


def load_product_cases(
    records_path: Path,
    expected_sha256: str,
    source_manifest_path: Path,
    train_manifest_path: Path,
    tree_manifest_path: Path,
    *,
    top_k: int,
    production: bool = True,
) -> tuple[tuple[PrivateCase, ...], ProductBaseline]:
    """Build an all-case shared pool, preserving labels only evaluator-side."""
    try:
        original = load_private_cases(
            records_path,
            expected_sha256,
            source_manifest_path,
            train_manifest_path,
            top_k=top_k,
            production=production,
        )
        payload = json.loads(records_path.read_text(encoding="utf-8"))
        records = payload["records"] if isinstance(payload, dict) else payload
        tree_manifest = json.loads(tree_manifest_path.read_text(encoding="utf-8"))
        entries = tree_manifest["cases"]
        if not isinstance(entries, dict) or set(entries) != {r["record_id"] for r in records}:
            _reject("failing tree coverage mismatch")
        lens = FailureLens.from_baseline()
        result_cases: list[PrivateCase] = []
        orders: dict[str, tuple[str, ...]] = {}
        for record, prior in zip(records, original, strict=True):
            root = verify_failing_tree(entries[record["record_id"]], record["source_commit"])
            index = lens.index(root)
            undecodable: list[str] = []
            for source_path, relative in iter_source_files(root, IndexConfig()):
                try:
                    source_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    undecodable.append(relative)
            if len(index.warnings) != len(undecodable) or any(
                not warning.startswith(f"Could not read {relative}:")
                for warning, relative in zip(index.warnings, undecodable, strict=True)
            ):
                _reject("unexpected product index warning")
            label_files = {item["location"]["file"] for item in record["candidates"]}
            mapped: dict[str, str] = {}
            for label_file in label_files:
                matches = [
                    path
                    for path in index.paths()
                    if path == label_file or path.endswith("/" + label_file)
                ]
                if len(matches) != 1:
                    _reject("ambiguous or missing labeled file")
                mapped[label_file] = matches[0]
            positive_files = frozenset(mapped[file] for file in prior.positive_file_ids)
            positive_functions = (
                frozenset(
                    f"{mapped[file]}::{name}"
                    for file, name in (item.rsplit("::", 1) for item in prior.positive_function_ids)
                )
                if prior.positive_function_ids is not None
                else None
            )
            positive_ids = frozenset(
                f"{mapped[file]}::{rest}"
                for file, rest in (item.split("::", 1) for item in prior.positive_ids)
            )
            for positive in positive_functions or ():
                file, name = positive.rsplit("::", 1)
                indexed = index.file(file)
                if (
                    indexed is None
                    or len([unit for unit in indexed.units if unit.name == name]) != 1
                ):
                    _reject("ambiguous or missing positive function span")
            raw_log = record["error_log"]
            parsed = lens.parsers.parse(raw_log)
            capacity = max(top_k, len(parsed.frames) if parsed is not None else 1)
            result = lens.diagnose(root, error_text=raw_log, top_k=capacity, index=index)
            if result.analysis_method != "baseline-stacktrace-lexical" or result.status not in {
                AnalysisStatus.OK,
                AnalysisStatus.NO_CANDIDATES,
                AnalysisStatus.INSUFFICIENT_EVIDENCE,
            }:
                _reject("invalid product retrieval")
            if result.status != AnalysisStatus.OK and result.candidates:
                _reject("invalid product retrieval")
            verify_failing_tree(entries[record["record_id"]], record["source_commit"])
            candidates = tuple(_product_candidate(item, root, index) for item in result.candidates)
            ids = tuple(item.id for item in candidates)
            if len(ids) != len(set(ids)) or len(ids) > capacity:
                _reject("ambiguous product retrieval")
            query = _query(raw_log, record.get("stack_trace"))
            request = RankRequest(
                query, candidates, top_k, _request_digest(query, candidates, top_k)
            )
            if request.digest in orders and orders[request.digest] != ids:
                _reject("conflicting product retrieval")
            orders[request.digest] = ids
            result_cases.append(
                PrivateCase(
                    prior.repository_id,
                    request,
                    positive_ids,
                    positive_files,
                    positive_functions,
                )
            )
        return tuple(result_cases), ProductBaseline(orders)
    except PrivateInputError:
        raise
    except Exception:
        _reject("invalid product pool input")
