"""Load the private v0.2 corpus as Stage 2 training and validation examples.

Reads ``train/records.json`` and ``dev/records.json`` from the private corpus
root, verifies the corpus digest and the pinned source clones, and rebuilds the
positive (mutated) and negative (clean) function texts from the recorded exact
mutation recipes. Train records become ``train`` examples and dev records become
the ``val`` examples used for validation loss and model selection. The Gate C
hold-out is never read here.

Aggregate counts only are returned by ``describe``; no record content is printed.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.corpus.build import canonical_digest, sha256_file
from training.corpus.mutations import Mutation, apply_mutation
from training.corpus.repos import HOLDOUT_REPOSITORIES, get_repo
from training.data_stage2 import Stage2Example

DEFAULT_CORPUS_ROOT: Path = Path.home() / ".local/share/failurelens/v02-corpus"
DEFAULT_V03_CORPUS_ROOT: Path = Path.home() / ".local/share/failurelens/v03-corpus"
PLACEHOLDER_DIGEST = "PENDING_CORPUS_BUILD"


def _function_lines(source: str, name: str, start: int, end: int) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
            and node.lineno == start
            and node.end_lineno is not None
        ):
            lines = source.splitlines(keepends=True)
            return "".join(lines[node.lineno - 1 : node.end_lineno])
    raise ValueError(f"function {name!r} not found at line {start}-{end}")


def _clean_source(root: Path, record: dict[str, Any], file: str) -> str:
    repo = get_repo(record["source_repository"])
    return (root / "sources" / repo.key / file).read_text(encoding="utf-8")


def verify_corpus(root: Path, expected_sha256: str) -> dict[str, str]:
    """Check the combined digest and every pinned clone; return the per-split digests."""
    if not expected_sha256 or expected_sha256 == PLACEHOLDER_DIGEST:
        raise ValueError("corpus digest is not set; build the corpus and fill the config")
    digests = {
        "train": sha256_file(root / "train" / "records.json"),
        "dev": sha256_file(root / "dev" / "records.json"),
    }
    actual = canonical_digest(digests)
    if actual != expected_sha256:
        raise ValueError(
            f"Corpus digest mismatch: expected {expected_sha256!r} but got {actual!r}. "
            "Use the authorised private corpus only."
        )
    return digests


def _verify_clones(root: Path, records: list[dict[str, Any]]) -> None:
    import subprocess

    seen: set[str] = set()
    for record in records:
        slug = record["source_repository"]
        if slug in seen:
            continue
        seen.add(slug)
        if slug in HOLDOUT_REPOSITORIES:
            raise ValueError("hold-out repository found in the training corpus")
        repo = get_repo(slug)
        clone = root / "sources" / repo.key
        head = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if head != record["source_commit"] or head != repo.commit:
            raise ValueError(f"Pinned clone for {repo.key!r} does not match the corpus record")


def _examples(root: Path, split_dir: str, split: str, dataset_version: str) -> list[Stage2Example]:
    records: list[dict[str, Any]] = json.loads(
        (root / split_dir / "records.json").read_text(encoding="utf-8")
    )
    recipes: dict[str, dict[str, Any]] = json.loads(
        (root / split_dir / "mutations.json").read_text(encoding="utf-8")
    )
    _verify_clones(root, records)
    examples: list[Stage2Example] = []
    for record in records:
        recipe = recipes[record["record_id"]]
        mutation = Mutation(
            function=recipe["function"],
            kind=recipe["kind"],
            line_no=recipe["line_no"],
            old=recipe["old"],
            new=recipe["new"],
            in_place=recipe["in_place"],
        )
        location = record["injection_location"]
        clean = _clean_source(root, record, location["file"])
        mutated = apply_mutation(clean, mutation)
        shift = 0 if mutation.in_place else 1
        positive = _function_lines(
            mutated, location["function"], location["start_line"], location["end_line"] + shift
        )
        negative_candidate = next(c for c in record["candidates"] if c["label"] == "negative")
        loc = negative_candidate["location"]
        negative = _function_lines(
            _clean_source(root, record, loc["file"]),
            loc["function"],
            loc["start_line"],
            loc["end_line"],
        )
        examples.append(
            Stage2Example(
                record_id=record["record_id"],
                query=record["error_log"].strip(),
                positive=positive,
                negative=negative,
                source_repository=record["source_repository"],
                dataset_version=dataset_version,
                split=split,
            )
        )
    return examples


def load_v02_examples(
    root: Path | None = None, *, expected_sha256: str, version: str = "v02"
) -> tuple[list[Stage2Example], list[Stage2Example]]:
    """Return ``(train_examples, val_examples)`` for a digest-pinned private corpus.

    ``version`` selects the corpus family (``"v02"`` or ``"v03"``); both share the same
    on-disk layout and recipe format.
    """
    default = DEFAULT_V03_CORPUS_ROOT if version == "v03" else DEFAULT_CORPUS_ROOT
    base = root if root is not None else default
    verify_corpus(base, expected_sha256)
    train = _examples(base, "train", "train", f"failurelens-train-{version}")
    val = _examples(base, "dev", "val", f"failurelens-dev-{version}")
    train_repos = {e.source_repository for e in train}
    val_repos = {e.source_repository for e in val}
    if train_repos & val_repos:
        raise ValueError("train and dev share a source repository")
    if len(train) < 30 or len(val) < 10:
        raise ValueError(f"corpus too small: {len(train)} train, {len(val)} dev")
    return train, val


def describe(train: list[Stage2Example], val: list[Stage2Example]) -> dict[str, Any]:
    """Aggregate, content-free summary suitable for logs and reports."""
    return {
        "train": len(train),
        "val": len(val),
        "train_by_repository": dict(Counter(e.source_repository for e in train)),
        "val_by_repository": dict(Counter(e.source_repository for e in val)),
    }


def load_v03_examples(
    root: Path | None = None, *, expected_sha256: str
) -> tuple[list[Stage2Example], list[Stage2Example]]:
    """Return ``(train_examples, val_examples)`` for the digest-pinned v0.3 corpus."""
    return load_v02_examples(root, expected_sha256=expected_sha256, version="v03")
