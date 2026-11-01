from __future__ import annotations

import pytest

from failurelens.split_leakage import SplitLeak, find_split_leaks, normalize_repository


def _record(record_id: str, repository: str) -> dict[str, str]:
    return {"record_id": record_id, "source_repository": repository}


def test_disjoint_repositories_have_no_leaks() -> None:
    splits = {
        "train": [_record("a-1", "https://github.com/org/alpha")],
        "eval": [_record("b-1", "https://github.com/org/beta")],
    }
    assert find_split_leaks(splits) == []


def test_repository_in_two_splits_is_a_leak() -> None:
    splits = {
        "train": [_record("a-1", "https://github.com/org/alpha")],
        "eval": [_record("a-2", "https://github.com/org/alpha")],
    }
    assert find_split_leaks(splits) == [
        SplitLeak("repository", "https://github.com/org/alpha", ("eval", "train"))
    ]


def test_repository_variants_compare_equal() -> None:
    splits = {
        "train": [_record("a-1", "https://github.com/Org/Alpha.git")],
        "eval": [_record("a-2", "https://github.com/org/alpha/")],
    }
    [leak] = find_split_leaks(splits)
    assert leak.kind == "repository"
    assert leak.key == "https://github.com/org/alpha"
    assert leak.splits == ("eval", "train")


def test_repeated_record_id_across_splits_is_reported() -> None:
    splits = {
        "train": [_record("same-id", "https://github.com/org/alpha")],
        "eval": [_record("same-id", "https://github.com/org/beta")],
    }
    leaks = find_split_leaks(splits)
    assert [leak.kind for leak in leaks] == ["record-id"]
    assert leaks[0].key == "same-id"


def test_repository_leaks_are_listed_before_record_id_leaks() -> None:
    splits = {
        "train": [_record("x", "https://github.com/org/alpha")],
        "eval": [
            _record("x", "https://github.com/org/alpha"),
        ],
    }
    kinds = [leak.kind for leak in find_split_leaks(splits)]
    assert kinds == ["repository", "record-id"]


def test_within_one_split_repeats_are_not_leaks() -> None:
    splits = {
        "train": [
            _record("a-1", "https://github.com/org/alpha"),
            _record("a-2", "https://github.com/org/alpha"),
        ],
        "eval": [],
    }
    assert find_split_leaks(splits) == []


def test_missing_repository_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing source_repository"):
        find_split_leaks({"train": [{"record_id": "x"}]})


def test_blank_repository_fails_closed() -> None:
    with pytest.raises(ValueError, match="source_repository is empty"):
        find_split_leaks({"train": [_record("x", "  /  ")]})


def test_normalize_repository_strips_and_folds() -> None:
    assert (
        normalize_repository("  https://GitHub.com/Org/Repo.GIT/ ") == "https://github.com/org/repo"
    )
