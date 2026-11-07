"""Tests for the Gate C v2 evaluation-set builder using only synthetic data."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest
from training.corpus import gatec_v2
from training.corpus.gatec_v2 import (
    CATEGORY_ABSENT,
    CATEGORY_INNERMOST,
    CATEGORY_MID,
    EVAL_REPOSITORIES,
    EvalCase,
    apply_edit,
    classify_surfacing,
    enumerate_extra_mutations,
    select_cases,
    validate_eval_table,
    write_train_manifest,
)
from training.corpus.mutations import Mutation, enumerate_functions
from training.corpus.repos import HOLDOUT_REPOSITORIES, REPOSITORIES, SourceRepo

SOURCE = textwrap.dedent(
    '''\
    def combine(items, sep):
        """Join items."""
        total = len(items)
        label = sep.join(items)
        count = total + len(label)
        items.append(label)
        return label, count
    '''
)


def test_eval_table_is_valid_and_disjoint() -> None:
    validate_eval_table()
    slugs = {r.slug.lower() for r in EVAL_REPOSITORIES}
    assert len(EVAL_REPOSITORIES) >= 4
    assert not slugs & {r.slug.lower() for r in REPOSITORIES}
    assert not slugs & {s.lower() for s in HOLDOUT_REPOSITORIES}
    assert not slugs & {s.lower() for s in gatec_v2.LEGACY_TRAIN_REPOSITORIES}
    assert all(r.role == "eval" and len(r.commit) == 40 for r in EVAL_REPOSITORIES)


def test_eval_table_rejects_overlap_and_bad_pins() -> None:
    first = EVAL_REPOSITORIES[0]
    with pytest.raises(ValueError, match="overlaps"):
        validate_eval_table(extra_train=(first.slug,))
    clash = SourceRepo(**{**first.__dict__, "slug": "pallets/click", "key": "zzz"})
    with pytest.raises(ValueError, match="overlaps"):
        validate_eval_table((clash,))
    held = SourceRepo(**{**first.__dict__, "slug": "psf/requests", "key": "zzz"})
    with pytest.raises(ValueError, match="overlaps"):
        validate_eval_table((held,))
    bad = SourceRepo(**{**first.__dict__, "commit": "abc"})
    with pytest.raises(ValueError, match="pin"):
        validate_eval_table((bad,))
    gpl = SourceRepo(**{**first.__dict__, "license": "GPL-3.0"})
    with pytest.raises(ValueError, match="license"):
        validate_eval_table((gpl,))


def test_extra_mutations_are_exact_parse_and_bounded() -> None:
    (info,) = enumerate_functions(SOURCE)
    mutations = enumerate_extra_mutations(SOURCE, info)
    kinds = {m.kind for m in mutations}
    assert {gatec_v2.KIND_SWAP, gatec_v2.KIND_DUP, gatec_v2.KIND_PASS} <= kinds
    for mutation in mutations:
        assert SOURCE.count(mutation.old) == 1
        edited = apply_edit(SOURCE, mutation)
        ast.parse(edited)
        delta = len(edited.splitlines()) - len(SOURCE.splitlines())
        assert delta in {0, 1}
        assert edited != SOURCE
    swaps = [m for m in mutations if m.kind == gatec_v2.KIND_SWAP]
    assert swaps and all("\n" in m.old and m.in_place for m in swaps)
    assert all(m.kind != gatec_v2.KIND_DUP or not m.in_place for m in mutations)


def test_argument_swap() -> None:
    source = "def f(a, b):\n    x = 1\n    y = max(a, b)\n    return x + y\n"
    (info,) = enumerate_functions(source)
    swaps = [m for m in enumerate_extra_mutations(source, info) if m.kind == gatec_v2.KIND_ARGSWAP]
    assert [apply_edit(source, m).splitlines()[2] for m in swaps] == ["    y = max(b, a)"]


def test_apply_edit_is_exact_and_rejects_stale_text() -> None:
    mutation = Mutation(
        "f", gatec_v2.KIND_SWAP, 3, "    a = 1\n    b = a", "    b = a\n    a = 1", True
    )
    source = 'def f():\n    """d"""\n    a = 1\n    b = a\n    return b\n'
    assert apply_edit(source, mutation).splitlines()[2:4] == ["    b = a", "    a = 1"]
    stale = Mutation("f", gatec_v2.KIND_SWAP, 3, "    a = 2\n    b = a", "x", True)
    with pytest.raises(ValueError, match="does not match"):
        apply_edit(source, stale)
    single = Mutation("f", "k", 3, "    a = 1", "    a = 2", True)
    assert apply_edit(source, single).splitlines()[2] == "    a = 2"


TRACE_INNER = (
    "Traceback (most recent call last):\n"
    '  File "tests/test_x.py", line 4, in test_it\n    run()\n'
    '  File "pkg/core.py", line 10, in run\n    helper()\n'
    '  File "pkg/util.py", line 3, in helper\n    raise ValueError\nValueError\n'
)


def test_surfacing_categories_from_stack_text_only() -> None:
    assert classify_surfacing(TRACE_INNER, "pkg/util.py", "helper") == CATEGORY_INNERMOST
    assert classify_surfacing(TRACE_INNER, "pkg/core.py", "run") == CATEGORY_MID
    assert classify_surfacing(TRACE_INNER, "pkg/other.py", "elsewhere") == CATEGORY_ABSENT
    assertion = (
        "Traceback (most recent call last):\n"
        '  File "pkg/util.py", line 3, in helper\n    x\n'
        '  File "tests/test_x.py", line 9, in test_it\n    assert 0\nAssertionError\n'
    )
    # root appears, but the failure surfaces afterwards in the test frame
    assert classify_surfacing(assertion, "pkg/util.py", "helper") == CATEGORY_MID


def _case(repo: str, index: int, category: str, kind: str, function: str) -> EvalCase:
    return EvalCase(
        {"record_id": f"{repo}-{index}"}, {"old": "a", "new": "b"}, category, kind, repo, function
    )


def test_selection_is_deterministic_stratified_and_balanced() -> None:
    cases = [
        _case(repo, i, category, kind, f"{repo}-fn{i // 2}")
        for repo in ("a/one", "b/two", "c/three", "d/four")
        for i, (category, kind) in enumerate(
            [(c, k) for c in (CATEGORY_INNERMOST, CATEGORY_MID, CATEGORY_ABSENT) for k in "wxyz"]
        )
    ]
    first = select_cases(cases, 20, seed=1)
    again = select_cases(list(reversed(cases)), 20, seed=1)
    assert [c.record["record_id"] for c in first] == [c.record["record_id"] for c in again]
    assert len(first) == 20
    repos = [c.repo for c in first]
    assert {repos.count(r) for r in set(repos)} == {5}
    categories = [c.category for c in first]
    assert categories.count(CATEGORY_MID) >= categories.count(CATEGORY_ABSENT) >= 4
    assert all(
        sum(c.function_key == f for c in first) <= 2 for f in {c.function_key for c in first}
    )


def test_selection_redistributes_shortfall() -> None:
    cases = [_case("a/one", i, CATEGORY_MID, "k", f"f{i}") for i in range(2)]
    cases += [_case("b/two", i, CATEGORY_MID, "k", f"g{i}") for i in range(12)]
    picked = select_cases(cases, 10, seed=3)
    assert len(picked) == 10
    assert sum(c.repo == "a/one" for c in picked) == 2


def test_train_manifest_is_regenerable_and_rejects_overlap(tmp_path: Path) -> None:
    eval_repos = ["tkem/cachetools", "python-validators/validators"]
    path = write_train_manifest(tmp_path, ["pallets/click", "extra/new"], eval_repos)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["training_repositories"] == ["extra/new", "pallets/click"]
    assert data["approval_id"]
    with pytest.raises(RuntimeError, match="overlap"):
        write_train_manifest(tmp_path, ["PYTHON-VALIDATORS/validators"], eval_repos)
    with pytest.raises(RuntimeError, match="overlap"):
        write_train_manifest(tmp_path, ["a/b"], ["psf/requests"])
    assert gatec_v2.default_train_repositories() == sorted(
        {r.slug for r in REPOSITORIES} | set(gatec_v2.LEGACY_TRAIN_REPOSITORIES)
    )
