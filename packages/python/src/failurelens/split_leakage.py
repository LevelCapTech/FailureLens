"""Non-leaking split checks for dataset records.

Records are split by repository, and variants of one bug share their source
repository and commit, so a repository that appears in two splits is leakage:
the same project or fix would let an evaluation split be learned from during
development. A repeated ``record_id`` across splits is reported the same way.
See the "Non-leaking splits" section of ``docs/data-contract.md``.

Like :mod:`failurelens.quality`, this module is pure and dependency-free. A record
without a usable ``source_repository`` raises ``ValueError`` instead of being
skipped, because a silently skipped record could hide leakage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["SplitLeak", "find_split_leaks", "normalize_repository"]


@dataclass(frozen=True)
class SplitLeak:
    """A repository or record id that appears in more than one split.

    ``splits`` holds the sorted names of every split the key appears in.
    """

    kind: Literal["repository", "record-id"]
    key: str
    splits: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping for reporting."""
        return {"kind": self.kind, "key": self.key, "splits": list(self.splits)}


def normalize_repository(value: str) -> str:
    """Return a comparison key for a repository identifier.

    Trims whitespace and trailing slashes, drops a ``.git`` suffix, and
    case-folds, so ``https://github.com/Org/Repo.git`` and
    ``https://github.com/org/repo`` compare equal. Raises ``ValueError`` when
    nothing is left.
    """
    key = value.strip().rstrip("/")
    if key.casefold().endswith(".git"):
        key = key[:-4]
    key = key.casefold()
    if not key:
        raise ValueError("source_repository is empty")
    return key


def find_split_leaks(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[SplitLeak]:
    """Return every repository and record id shared by two or more splits.

    Args:
        splits: Mapping of split name (for example ``"train"`` or ``"eval"``) to
            its records. Records are not mutated.

    Returns:
        Repository leaks first, then record-id leaks, each sorted by key. An
        empty list means no cross-split leakage was found.

    Raises:
        ValueError: A record has no usable ``source_repository``.
    """
    repository_splits: dict[str, set[str]] = {}
    record_splits: dict[str, set[str]] = {}
    for split_name, records in splits.items():
        for index, record in enumerate(records):
            repository = record.get("source_repository")
            if not isinstance(repository, str):
                raise ValueError(f"split {split_name!r} record {index}: missing source_repository")
            try:
                key = normalize_repository(repository)
            except ValueError as exc:
                raise ValueError(f"split {split_name!r} record {index}: {exc}") from exc
            repository_splits.setdefault(key, set()).add(split_name)
            record_id = record.get("record_id")
            if record_id is not None:
                record_splits.setdefault(str(record_id), set()).add(split_name)

    leaks = [
        SplitLeak("repository", key, tuple(sorted(names)))
        for key, names in sorted(repository_splits.items())
        if len(names) > 1
    ]
    leaks += [
        SplitLeak("record-id", key, tuple(sorted(names)))
        for key, names in sorted(record_splits.items())
        if len(names) > 1
    ]
    return leaks
