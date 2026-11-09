"""Tests for the Gate C v3 evaluation-set builder using only synthetic data."""

from __future__ import annotations

import inspect
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from training.corpus import gatec_v2, gatec_v3
from training.corpus.gatec_v2 import EvalCase
from training.corpus.gatec_v3 import (
    CATEGORIES,
    CATEGORY_ABSENT_RELATED,
    CATEGORY_ABSENT_UNRELATED,
    V3_REPOSITORIES,
    _static_graph,
    classify_v3,
    default_train_repositories,
    select_v3,
    validate_v3_table,
)
from training.corpus.repos import HOLDOUT_REPOSITORIES, REPOSITORIES, SourceRepo


def test_v3_table_is_valid_and_disjoint_from_everything_used() -> None:
    validate_v3_table()
    slugs = {r.slug.lower() for r in V3_REPOSITORIES}
    assert len(V3_REPOSITORIES) >= 5
    assert not slugs & {r.slug.lower() for r in REPOSITORIES}
    assert not slugs & {s.lower() for s in HOLDOUT_REPOSITORIES}
    assert not slugs & {s.lower() for s in gatec_v2.LEGACY_TRAIN_REPOSITORIES}
    assert not slugs & {r.slug.lower() for r in gatec_v2.EVAL_REPOSITORIES}
    assert all(r.role == "eval" and len(r.commit) == 40 for r in V3_REPOSITORIES)


@pytest.mark.parametrize(
    "slug",
    ["tkem/cachetools", "psf/requests", "pallets/click", "python-humanize/humanize"],
)
def test_v3_table_rejects_any_earlier_source(slug: str) -> None:
    clash = SourceRepo(**{**V3_REPOSITORIES[0].__dict__, "slug": slug, "key": "zzz"})
    with pytest.raises(ValueError, match="overlaps"):
        validate_v3_table((clash,))
    with pytest.raises(ValueError, match="overlaps"):
        validate_v3_table(extra_train=(V3_REPOSITORIES[0].slug,))


def test_v3_table_rejects_bad_pin_license_and_role() -> None:
    first = V3_REPOSITORIES[0]
    for change in ({"commit": "abc"}, {"license": "GPL-3.0"}, {"role": "train"}):
        bad = SourceRepo(**{**first.__dict__, **change})
        with pytest.raises(ValueError):
            validate_v3_table((bad,))


def _tree(tmp_path: Path) -> tuple[dict[tuple[str, str], set[str]], Counter[str]]:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        textwrap.dedent(
            """\
            def entry(x):
                return helper_one(x) + 1

            def helper_one(x):
                return inner_two(x)

            def inner_two(x):
                return x

            def lonely(x):
                return x * 2

            def get(x):
                return x

            def user_a(x):
                return get(x)
            """
        ),
        encoding="utf-8",
    )
    _static_graph.cache_clear()
    return _static_graph(str(tmp_path), ("pkg/mod.py",))


def _trace(*frames: tuple[str, str]) -> str:
    lines = ["Traceback (most recent call last):"]
    for path, name in frames:
        lines += [f'  File "{path}", line 3, in {name}', "    x"]
    return "\n".join(lines) + "\nValueError: boom"


def test_categories_use_trace_and_a_unique_name_call_check(tmp_path: Path) -> None:
    edges, defs = _tree(tmp_path)
    f = "pkg/mod.py"
    inner = _trace((f, "entry"), (f, "helper_one"))
    assert classify_v3(inner, f, "helper_one", edges, defs) == gatec_v3.CATEGORY_INNERMOST
    # absent from the stack but called by a production frame (unique name)
    assert classify_v3(inner, f, "inner_two", edges, defs) == CATEGORY_ABSENT_RELATED
    # absent and no direct relation to the production frames
    unrelated = _trace((f, "entry"))
    assert classify_v3(unrelated, f, "lonely", edges, defs) == CATEGORY_ABSENT_UNRELATED
    # a non-unique name must not create a relation
    trace = _trace((f, "user_a"))
    assert classify_v3(trace, f, "get", edges, defs) == CATEGORY_ABSENT_RELATED
    edges2 = {**edges, ("pkg/other.py", "get"): set()}
    defs2 = Counter(defs)
    defs2["get"] += 1
    assert classify_v3(trace, f, "get", edges2, defs2) == CATEGORY_ABSENT_UNRELATED


def test_test_frames_never_create_a_relation(tmp_path: Path) -> None:
    edges, defs = _tree(tmp_path)
    trace = _trace(("tests/test_mod.py", "test_it"))
    assert classify_v3(trace, "pkg/mod.py", "inner_two", edges, defs) == CATEGORY_ABSENT_UNRELATED


def _case(repo: str, n: int, category: str, kind: str) -> EvalCase:
    name = f"f{n}"
    return EvalCase(
        {"record_id": f"{repo}-{n:04d}", "source_repository": repo},
        {"file": "m.py", "function": name, "old": "a", "new": "b", "in_place": True},
        category,
        kind,
        repo,
        f"m.py::{name}",
    )


def test_selection_is_deterministic_and_covers_all_four_categories() -> None:
    cases = [
        _case(repo, i, CATEGORIES[i % 4], f"kind{i % 5}")
        for repo in ("a/one", "b/two", "c/three", "d/four", "e/five")
        for i in range(24)
    ]
    first = select_v3(cases, 40, seed=7)
    second = select_v3(list(reversed(cases)), 40, seed=7)
    assert [c.record["record_id"] for c in first] == [c.record["record_id"] for c in second]
    assert len(first) == 40
    assert set(Counter(c.category for c in first)) == set(CATEGORIES)
    per_repo = Counter(c.repo for c in first)
    assert max(per_repo.values()) - min(per_repo.values()) <= 1
    counts = Counter(c.category for c in first)
    assert counts[CATEGORY_ABSENT_RELATED] == max(counts.values())


def test_selection_reports_shortfall_instead_of_padding() -> None:
    cases = [_case("a/one", i, CATEGORIES[0], "k") for i in range(5)]
    assert len(select_v3(cases, 40, seed=1)) <= 5


def test_default_train_repositories_cover_train_dev_and_legacy() -> None:
    repos = set(default_train_repositories())
    assert {"pallets/click", "python-attrs/attrs", "python-humanize/humanize"} <= repos
    assert not repos & {r.slug for r in V3_REPOSITORIES}


def test_train_manifest_is_regenerable_and_rejects_overlap(tmp_path: Path) -> None:
    eval_repos = [r.slug for r in V3_REPOSITORIES]
    path = gatec_v2.write_train_manifest(tmp_path, default_train_repositories(), eval_repos)
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert not set(data["training_repositories"]) & set(eval_repos)
    with pytest.raises(RuntimeError, match="overlap"):
        gatec_v2.write_train_manifest(
            tmp_path, [*default_train_repositories(), eval_repos[0]], eval_repos
        )


def test_generate_parameters_do_not_change_v2_defaults() -> None:
    params = inspect.signature(gatec_v2.generate_eval_repo).parameters
    assert params["hard"].default is False
    assert params["dataset_id"].default == gatec_v2.EVAL_DATASET_ID
    assert params["categorize"].default is None
