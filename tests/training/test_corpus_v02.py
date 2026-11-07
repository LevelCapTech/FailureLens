"""Tests for the v0.2 corpus generator, using a tiny synthetic project only."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from evaluation.private_adapter import load_private_cases
from evaluation.product_pool import load_product_cases
from training.corpus import build, loader
from training.corpus.mutations import (
    Mutation,
    apply_mutation,
    enumerate_functions,
    enumerate_mutations,
)
from training.corpus.repos import (
    HOLDOUT_REPOSITORIES,
    REPOSITORIES,
    SourceRepo,
    validate_table,
)
from training.corpus.runner import extract_failure_log, terminal_exception

MODULE = textwrap.dedent(
    '''\
    """Tiny text utilities."""


    def clamp(value, low, high):
        """Clamp value into [low, high]."""
        if low > high:
            raise ValueError("low must not exceed high")
        if value < low:
            return low
        if value > high:
            return high
        return value


    def is_even(n):
        remainder = n % 2
        return remainder == 0


    def chunk(items, size):
        result = []
        for start in range(0, len(items), size):
            result.append(items[start : start + size])
        return result


    def count_vowels(text):
        total = 0
        for ch in text.lower():
            if ch in "aeiou":
                total += 1
        return total


    def unused_helper(x):
        y = x + 1
        z = y * 2
        return z - 1
    '''
)

TESTS = textwrap.dedent(
    """\
    import pytest
    from mini.textutil import chunk, clamp, count_vowels, is_even


    @pytest.mark.parametrize(
        ("value", "expected"), [(-5, 0), (0, 0), (3, 3), (10, 10), (11, 10), (99, 10)]
    )
    def test_clamp(value, expected):
        assert clamp(value, 0, 10) == expected


    @pytest.mark.parametrize(("n", "expected"), [(2, True), (3, False), (0, True), (7, False)])
    def test_is_even(n, expected):
        assert is_even(n) is expected


    def test_chunk():
        assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


    def test_chunk_exact():
        assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


    def test_vowels():
        assert count_vowels("Education") == 5
    """
)


def test_table_is_valid_and_disjoint_from_holdout() -> None:
    validate_table()
    slugs = {r.slug for r in REPOSITORIES}
    assert not slugs & HOLDOUT_REPOSITORIES
    assert {r.role for r in REPOSITORIES} == {"train", "dev"}
    assert len([r for r in REPOSITORIES if r.role == "dev"]) >= 3
    assert len([r for r in REPOSITORIES if r.role == "train"]) >= 4


def test_table_rejects_holdout_and_bad_pins() -> None:
    bad = SourceRepo("psf/requests", "r", "a" * 40, "MIT", "dev", ("r/",), (), ("t",), ".", "L")
    with pytest.raises(ValueError, match="hold-out"):
        validate_table((bad,))
    short = SourceRepo("o/n", "n", "abc", "MIT", "dev", ("n/",), (), ("t",), ".", "L")
    with pytest.raises(ValueError, match="pin"):
        validate_table((short,))


def test_classify_license() -> None:
    mit = "Permission is hereby granted, free of charge, to any person obtaining a copy"
    bsd3 = (
        "Redistribution and use in source and binary forms, with or without modification. "
        "Neither the name of the copyright holder nor the names of its contributors"
    )
    assert build.classify_license(mit) == "MIT"
    assert build.classify_license(bsd3) == "BSD-3-Clause"
    variant = (
        "Redistribution and use in source and binary forms, with or without modification. "
        "The names of the contributors may not be used to endorse or promote products "
        "derived from this software"
    )
    assert build.classify_license(variant) == "BSD-3-Clause"
    assert build.classify_license("All rights reserved.") is None


def test_enumerate_functions_filters() -> None:
    names = [f.name for f in enumerate_functions(MODULE)]
    assert names == ["clamp", "is_even", "chunk", "count_vowels", "unused_helper"]
    duplicated = (
        "def f(a):\n    x = 1\n    return a\n\n\nclass C:\n    def f(self):\n        return 1\n"
    )
    assert enumerate_functions(duplicated) == []  # ambiguous name is not a fault site
    assert enumerate_functions("def broken(:\n") == []


def test_mutations_are_exact_unique_and_parse() -> None:
    kinds: set[str] = set()
    for info in enumerate_functions(MODULE):
        lines = MODULE.splitlines()
        function_text = "\n".join(lines[info.start_line - 1 : info.end_line])
        for mutation in enumerate_mutations(MODULE, info):
            kinds.add(mutation.kind)
            assert function_text.count(mutation.old) == 1
            assert mutation.old in MODULE
            mutated = apply_mutation(MODULE, mutation)
            ast.parse(mutated)
            assert mutated != MODULE
            delta = len(mutated.splitlines()) - len(lines)
            assert delta == mutation.inserted_lines
    assert {
        "comparison-operator flip",
        "constant off-by-one",
        "arithmetic-operator flip",
        "early-return None",
        "raised exception",
        "wrong-value return",
    } <= kinds


def test_boolean_and_negation_and_membership_mutations() -> None:
    source = (
        "def check(a, b, items):\n"
        "    if not a and b in items:\n"
        "        return True\n"
        "    if a is None or b:\n"
        "        return False\n"
        "    return a >= b\n"
    )
    info = enumerate_functions(source)[0]
    produced = {(m.kind, m.new.strip()) for m in enumerate_mutations(source, info)}
    assert ("boolean constant flip", "return False") in produced
    assert ("negation removal", "if a and b in items:") in produced
    assert ("boolean-operator flip", "if not a or b in items:") in produced
    assert ("comparison-operator flip", "if not a and b not in items:") in produced
    assert ("comparison-operator flip", "if a is not None or b:") in produced
    assert ("comparison-operator flip", "return a > b") in produced


def test_apply_mutation_rejects_stale_line() -> None:
    info = enumerate_functions(MODULE)[0]
    mutation = enumerate_mutations(MODULE, info)[0]
    stale = Mutation(**{**mutation.__dict__, "old": "something else"})
    with pytest.raises(ValueError):
        apply_mutation(MODULE, stale)


def test_extract_failure_log_relativizes_paths(tmp_path: Path) -> None:
    stdout = (
        "F\n\n=================================== FAILURES ===================================\n"
        "___________________________________ test_x ___________________________________\n"
        "Traceback (most recent call last):\n"
        f'  File "{tmp_path}/tests/test_x.py", line 3, in test_x\n    assert f() == 1\n'
        "AssertionError\n\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_x.py::test_x\n1 failed in 0.02s\n"
    )
    extracted = extract_failure_log(stdout, tmp_path)
    assert extracted is not None
    log, trace = extracted
    assert str(tmp_path) not in log
    assert 'File "tests/test_x.py"' in log
    assert trace.startswith("Traceback (most recent call last):")
    assert "short test summary" not in log and "0.02s" not in log
    assert terminal_exception(trace) == "AssertionError"
    assert extract_failure_log("no failures here", tmp_path) is None


def _synthetic(tmp_path: Path, role: str) -> tuple[build.Workspace, SourceRepo]:
    root = tmp_path / "private"
    clone = root / "sources" / "mini"
    (clone / "mini").mkdir(parents=True)
    (clone / "tests").mkdir()
    (clone / "mini" / "__init__.py").write_text("", encoding="utf-8")
    (clone / "mini" / "textutil.py").write_text(MODULE, encoding="utf-8")
    (clone / "tests" / "test_text.py").write_text(TESTS, encoding="utf-8")
    (clone / "LICENSE").write_text(
        "Permission is hereby granted, free of charge, to any person", encoding="utf-8"
    )

    def git(*args: str) -> str:
        done = subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@example.invalid",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    commit = git("rev-parse", "HEAD")
    repo = SourceRepo(
        "example/mini",
        "mini",
        commit,
        "MIT",
        role,
        ("mini/",),
        ("mini/__init__.py",),
        ("tests",),
        ".",
        "LICENSE",
    )
    ws = build.Workspace(root, Path(sys.executable))
    build.copy_tracked_files(clone, ws.work(repo))
    return ws, repo


def test_license_evidence_matches_declared_license(tmp_path: Path) -> None:
    ws, repo = _synthetic(tmp_path, "dev")
    evidence = build.license_evidence(ws.source(repo), repo)
    assert evidence["spdx"] == "MIT"
    wrong = SourceRepo(**{**repo.__dict__, "license": "BSD-3-Clause"})
    with pytest.raises(ValueError, match="license text"):
        build.license_evidence(ws.source(repo), wrong)


@pytest.fixture(scope="module")
def generated(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[build.Workspace, SourceRepo, list[build.GeneratedCase], dict[str, object]]:
    ws, repo = _synthetic(tmp_path_factory.mktemp("corpus"), "dev")
    cases, report = build.generate_for_repo(ws, repo, seed=7, target=6, per_function=2)
    return ws, repo, cases, report


def test_generation_produces_valid_pass_fail_records(
    generated: tuple[build.Workspace, SourceRepo, list[build.GeneratedCase], dict[str, object]],
) -> None:
    ws, repo, cases, report = generated
    assert report["baseline_passed_tests"] == 13
    assert len(cases) >= 3, report
    kept, validation = build.validate_cases(cases)
    assert len(kept) == len(cases), validation
    for case in cases:
        record = case.record
        assert record["pre_change_result"]["exit_code"] == 0
        assert record["post_change_result"]["exit_code"] == 1
        assert str(ws.root) not in record["error_log"]
        assert "Traceback (most recent call last):" in record["stack_trace"]
        labels = [c["label"] for c in record["candidates"]]
        assert labels[0] == "positive" and "negative" in labels
        # the work copy is restored byte for byte after every attempt
        assert (ws.work(repo) / case.mutation["file"]).read_text(encoding="utf-8") == MODULE


def test_dev_trees_load_through_evaluator_inputs(
    generated: tuple[build.Workspace, SourceRepo, list[build.GeneratedCase], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, repo, cases, _ = generated
    kept, _ = build.validate_cases(cases)
    kept = build._renumber(kept)
    out = ws.root / "dev"
    records = [c.record for c in kept]
    build.write_private_json(out / "records.json", records)
    build.write_dev_trees(ws, kept, out, resolve=lambda _slug: repo)
    build.write_private_json(
        out / "train_manifest.json",
        {"approval_id": "test-approval", "training_repositories": ["other/repo"]},
    )
    digest = build.sha256_file(out / "records.json")
    private = load_private_cases(
        out / "records.json",
        digest,
        out / "source_manifest.json",
        out / "train_manifest.json",
        top_k=5,
        production=False,
    )
    assert len(private) == len(kept)
    pooled, baseline = load_product_cases(
        out / "records.json",
        digest,
        out / "source_manifest.json",
        out / "train_manifest.json",
        out / "tree_manifest.json",
        top_k=5,
        production=False,
    )
    assert len(pooled) == len(kept) and baseline is not None
    # production mode must still demand the Gate-C-sized evidence base
    with pytest.raises(Exception, match="insufficient"):
        load_private_cases(
            out / "records.json",
            digest,
            out / "source_manifest.json",
            out / "train_manifest.json",
            top_k=5,
        )
    # a dev repository that is also in the training manifest is rejected
    build.write_private_json(
        out / "train_manifest.json",
        {"approval_id": "test-approval", "training_repositories": [repo.slug]},
    )
    with pytest.raises(Exception, match="overlap"):
        load_private_cases(
            out / "records.json",
            digest,
            out / "source_manifest.json",
            out / "train_manifest.json",
            top_k=5,
            production=False,
        )


def test_loader_rebuilds_positive_and_negative_texts(
    generated: tuple[build.Workspace, SourceRepo, list[build.GeneratedCase], dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, repo, cases, _ = generated
    kept = build._renumber(build.validate_cases(cases)[0])
    for split in ("train", "dev"):
        out = ws.root / split
        build.write_private_json(out / "records.json", [c.record for c in kept])
        build.write_private_json(
            out / "mutations.json", {c.record["record_id"]: c.mutation for c in kept}
        )
    monkeypatch.setattr(loader, "get_repo", lambda _slug: repo)
    examples = loader._examples(ws.root, "train", "train", "failurelens-train-v02")
    assert len(examples) == len(kept)
    for example, case in zip(examples, kept, strict=True):
        name = case.mutation["function"]
        assert example.positive.startswith(f"def {name}(")
        assert example.positive != loader._function_lines(MODULE, name, *_span(MODULE, name))
        assert "Traceback" in example.query and str(ws.root) not in example.query
        assert example.negative.startswith("def ") and name not in example.negative.splitlines()[0]
    digests = {
        "train": build.sha256_file(ws.root / "train" / "records.json"),
        "dev": build.sha256_file(ws.root / "dev" / "records.json"),
    }
    expected = build.canonical_digest(digests)
    assert loader.verify_corpus(ws.root, expected) == digests
    with pytest.raises(ValueError, match="mismatch"):
        loader.verify_corpus(ws.root, "0" * 64)
    with pytest.raises(ValueError, match="not set"):
        loader.verify_corpus(ws.root, loader.PLACEHOLDER_DIGEST)


def _span(source: str, name: str) -> tuple[int, int]:
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return node.lineno, node.end_lineno or node.lineno


def test_describe_is_content_free() -> None:
    summary = loader.describe([], [])
    assert json.dumps(summary) and summary["train"] == 0


def test_v02_candidate_configs_reference_corpus_by_digest_only() -> None:
    configs = sorted(Path("training/configs").glob("stage2_v02_*.json"))
    assert len(configs) >= 3
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in configs]
    for config in loaded:
        assert config["corpus"] == "failurelens-corpus-v02"
        digest = config["dataset_sha256"]
        assert digest == loader.PLACEHOLDER_DIGEST or len(digest) == 64
        assert "/" not in json.dumps(config).replace("nomic-ai/CodeRankEmbed", "")
        assert config["model_revision"] == "3c4b60807d71f79b43f3c4363786d9493691f8b1"
    # candidates must differ in at least one hyperparameter
    keys = ("learning_rate", "epochs", "batch_size", "temperature")
    assert len({tuple(c[k] for k in keys) for c in loaded}) == len(loaded)
