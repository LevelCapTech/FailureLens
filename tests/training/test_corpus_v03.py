"""Tests for the v0.3 hard mutation kinds and corpus profile, using a tiny synthetic project."""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from evaluation.private_adapter import load_private_cases
from evaluation.product_pool import load_product_cases
from training.corpus import build, loader
from training.corpus.mutations import (
    ALL_KINDS,
    BASE_KINDS,
    HARD_KINDS,
    KIND_BOUND,
    KIND_CALLEE,
    KIND_DEFAULT,
    KIND_DROP_ARG,
    KIND_GUARD,
    KIND_MULTI,
    KIND_SWAP_ARGS,
    FunctionInfo,
    Mutation,
    apply_mutation,
    enumerate_functions,
    enumerate_mutations,
)
from training.corpus.repos import REPOSITORIES, SourceRepo, validate_table
from training.finetune import _freeze_all_but_last_encoder_layer

MODULE = textwrap.dedent(
    '''\
    """Tiny sequence utilities."""


    def pad(items, width=3, fill=" "):
        if width < 0:
            raise ValueError("negative width")
        result = list(items)
        for _ in range(width - len(result)):
            result.append(fill)
        return result


    def window(items, size):
        out = []
        for start in range(len(items) - size + 1):
            out.append(items[start : start + size])
        return out


    def combine(a, b, scale):
        first = scale * a
        second = scale * b
        return first + second


    def span(low, high):
        return high - low


    def total_span(pairs):
        total = 0
        for low, high in pairs:
            total += span(low, high)
        return total


    def head(items, count):
        picked = items[:count]
        return picked


    def unused(x):
        y = x + 1
        z = y * 2
        return z - 1
    '''
)

TESTS = textwrap.dedent(
    """\
    import pytest
    from mini.seq import combine, head, pad, span, total_span, window


    @pytest.mark.parametrize(("n", "expected"), [(0, 3), (1, 3), (2, 3), (3, 3), (5, 5)])
    def test_pad_width(n, expected):
        assert len(pad(list(range(n)), expected)) == max(n, expected)


    def test_pad_default_width():
        assert pad([1]) == [1, " ", " "]


    def test_pad_negative():
        with pytest.raises(ValueError):
            pad([1], -1)


    def test_window():
        assert window([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]


    def test_window_full():
        assert window([1, 2], 2) == [[1, 2]]


    def test_combine():
        assert combine(1, 2, 3) == 9


    def test_span():
        assert span(2, 7) == 5


    def test_total_span():
        assert total_span([(1, 3), (0, 10)]) == 12


    def test_head():
        assert head([1, 2, 3, 4], 2) == [1, 2]


    def test_head_none():
        assert head([1, 2, 3], 0) == []
    """
)


def _function(source: str, name: str) -> ast.FunctionDef:
    return next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _all_hard_mutations() -> list[Mutation]:
    found: list[Mutation] = []
    for info in enumerate_functions(MODULE):
        found.extend(enumerate_mutations(MODULE, info, hard=True))
    return found


def test_kind_sets_are_consistent() -> None:
    assert set(ALL_KINDS) == set(BASE_KINDS) | HARD_KINDS
    assert not set(BASE_KINDS) & HARD_KINDS
    assert KIND_CALLEE in HARD_KINDS


def test_base_enumeration_has_no_hard_kinds() -> None:
    for info in enumerate_functions(MODULE):
        assert {m.kind for m in enumerate_mutations(MODULE, info)} <= set(BASE_KINDS)


def test_hard_kinds_are_enumerated_exact_and_keep_line_budget() -> None:
    mutations = _all_hard_mutations()
    kinds = {m.kind for m in mutations}
    assert {
        KIND_SWAP_ARGS,
        KIND_DROP_ARG,
        KIND_BOUND,
        KIND_DEFAULT,
        KIND_GUARD,
        KIND_MULTI,
    } <= kinds
    lines = MODULE.splitlines()
    for mutation in mutations:
        info = next(i for i in enumerate_functions(MODULE) if i.name == mutation.function)
        function_text = "\n".join(lines[info.start_line - 1 : info.end_line])
        assert function_text.count(mutation.old) == 1
        assert mutation.old != mutation.new
        assert mutation.inserted_lines in {0, 1}
        assert mutation.in_place == (mutation.inserted_lines == 0)
        mutated = apply_mutation(MODULE, mutation)
        ast.parse(mutated)
        assert mutated.count("\n") - MODULE.count("\n") == mutation.inserted_lines
        # exactly the recipe text changed, nothing else
        assert mutated.replace(mutation.new, mutation.old, 1) == MODULE


def test_multi_line_and_guard_recipes_span_several_lines() -> None:
    mutations = _all_hard_mutations()
    multi = [m for m in mutations if m.kind in {KIND_MULTI, KIND_GUARD}]
    assert multi and all(m.old_line_count >= 2 for m in multi)
    guard = next(m for m in multi if m.kind == KIND_GUARD)
    assert guard.new.splitlines()[-1].strip() == "pass"
    assert guard.new.splitlines()[0] == guard.old.splitlines()[0]


def test_default_and_bound_mutations_target_expected_sites() -> None:
    mutations = _all_hard_mutations()
    default = [m for m in mutations if m.kind == KIND_DEFAULT]
    assert any("width=4" in m.new for m in default)
    bound = [m for m in mutations if m.kind == KIND_BOUND]
    assert any("len(items) - size + 1 - 1" in m.new or "- 1" in m.new for m in bound)


def test_apply_mutation_rejects_stale_multi_line_text() -> None:
    mutation = next(m for m in _all_hard_mutations() if m.kind == KIND_GUARD)
    stale = MODULE.replace("negative width", "other message")
    with pytest.raises(ValueError, match="does not match"):
        apply_mutation(stale, mutation)


def test_classify_surfacing_relabels_only_callee_changes() -> None:
    repo = SourceRepo(
        "example/mini",
        "mini",
        "0" * 40,
        "MIT",
        "train",
        ("mini/",),
        ("mini/__init__.py",),
        ("tests",),
        ".",
        "LICENSE",
    )
    info = FunctionInfo("span", 1, 2, 2)
    mutation = Mutation("span", "constant off-by-one", 1, "a", "b", True)
    caller_trace = (
        "Traceback (most recent call last):\n"
        '  File "tests/test_seq.py", line 3, in test_total_span\n    total_span(x)\n'
        '  File "mini/seq.py", line 30, in total_span\n    total += span(low, high)\n'
        "TypeError: bad\n"
    )
    root_trace = caller_trace.replace("in total_span", "in span")
    test_only = (
        "Traceback (most recent call last):\n"
        '  File "tests/test_seq.py", line 3, in test_span\n    assert span(2, 7) == 5\n'
        "AssertionError\n"
    )
    previous = build.PROFILE
    try:
        build.set_profile("v03")
        relabelled, in_trace = build.classify_surfacing(repo, info, mutation, caller_trace)
        assert relabelled.kind == KIND_CALLEE and not in_trace
        assert relabelled.old == mutation.old and relabelled.new == mutation.new
        kept, in_trace = build.classify_surfacing(repo, info, mutation, root_trace)
        assert kept.kind == mutation.kind and in_trace
        kept, in_trace = build.classify_surfacing(repo, info, mutation, test_only)
        assert kept.kind == mutation.kind and not in_trace  # no frame inside the project
        build.set_profile("v02")
        kept, in_trace = build.classify_surfacing(repo, info, mutation, caller_trace)
        assert kept.kind == mutation.kind
    finally:
        build.PROFILE = previous


def test_profile_tables_and_repository_membership() -> None:
    validate_table()
    assert build.PROFILES["v02"].hard is False and build.PROFILES["v03"].hard is True
    assert build.PROFILES["v03"].corpus_id.endswith("v03")
    v03 = [r for r in REPOSITORIES if "v03" in r.profiles]
    v02 = [r for r in REPOSITORIES if "v02" in r.profiles]
    assert {r.slug for r in v02} < {r.slug for r in v03}
    assert sum(r.role == "train" for r in v03) > sum(r.role == "train" for r in v02)
    assert {r.slug for r in v03 if r.role == "dev"} == {r.slug for r in v02 if r.role == "dev"}


def _synthetic(tmp_path: Path, role: str) -> tuple[build.Workspace, SourceRepo]:
    root = tmp_path / "private"
    clone = root / "sources" / "mini"
    (clone / "mini").mkdir(parents=True)
    (clone / "tests").mkdir()
    (clone / "mini" / "__init__.py").write_text("", encoding="utf-8")
    (clone / "mini" / "seq.py").write_text(MODULE, encoding="utf-8")
    (clone / "tests" / "test_seq.py").write_text(TESTS, encoding="utf-8")
    (clone / "LICENSE").write_text(
        "Permission is hereby granted, free of charge, to any person", encoding="utf-8"
    )

    def run_git(*args: str) -> str:
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

    run_git("init", "-q")
    run_git("add", "-A")
    run_git("commit", "-q", "-m", "init")
    repo = SourceRepo(
        "example/mini",
        "mini",
        run_git("rev-parse", "HEAD"),
        "MIT",
        role,
        ("mini/",),
        ("mini/__init__.py",),
        ("tests",),
        ".",
        "LICENSE",
        ("v03",),
    )
    ws = build.Workspace(root, Path(sys.executable))
    build.copy_tracked_files(clone, ws.work(repo))
    return ws, repo


@pytest.fixture(scope="module")
def generated(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[build.Workspace, SourceRepo, list[build.GeneratedCase]]]:
    previous = build.PROFILE
    build.set_profile("v03")
    try:
        ws, repo = _synthetic(tmp_path_factory.mktemp("corpus03"), "dev")
        cases, report = build.generate_for_repo(ws, repo, seed=11, target=12, per_function=3)
        assert report["baseline_passed_tests"] >= 10
        yield ws, repo, cases
    finally:
        build.PROFILE = previous


def test_v03_generation_records_hard_cases(
    generated: tuple[build.Workspace, SourceRepo, list[build.GeneratedCase]],
) -> None:
    ws, repo, cases = generated
    assert len(cases) >= 4
    kept, validation = build.validate_cases(cases)
    # the duplicate gate may drop cases that fail the same test the same way
    assert len(kept) >= len(cases) - 2 and len(kept) >= 3, validation
    methods = {c.record["injection_method"] for c in cases}
    assert methods & HARD_KINDS, methods
    for case in cases:
        assert case.record["record_id"].startswith("v03-")
        assert case.record["dataset_id"] == "failurelens-dev-v03"
        assert case.record["generator_version"] == "corpus-v03-1.0"
        assert "root_in_trace" in case.mutation and "base_kind" in case.mutation
        assert case.record["post_change_result"]["exit_code"] == 1
        assert (ws.work(repo) / case.mutation["file"]).read_text(encoding="utf-8") == MODULE


def test_v03_dev_trees_load_through_evaluator_inputs(
    generated: tuple[build.Workspace, SourceRepo, list[build.GeneratedCase]],
) -> None:
    ws, repo, cases = generated
    previous = build.PROFILE
    build.set_profile("v03")
    try:
        # _renumber edits records in place; keep the shared fixture cases untouched
        kept = build._renumber(copy.deepcopy(build.validate_cases(cases)[0]))
    finally:
        build.PROFILE = previous
    assert all(c.record["record_id"].startswith("v03-mini-") for c in kept)
    out = ws.root / "dev"
    build.write_private_json(out / "records.json", [c.record for c in kept])
    build.write_dev_trees(ws, kept, out, resolve=lambda _slug: repo)
    build.write_private_json(
        out / "train_manifest.json",
        {"approval_id": "test-approval", "training_repositories": ["other/repo"]},
    )
    digest = build.sha256_file(out / "records.json")
    args = (
        out / "records.json",
        digest,
        out / "source_manifest.json",
        out / "train_manifest.json",
    )
    assert len(load_private_cases(*args, top_k=5, production=False)) == len(kept)
    pooled, baseline = load_product_cases(
        *args, out / "tree_manifest.json", top_k=5, production=False
    )
    assert len(pooled) == len(kept) and baseline is not None


def test_v03_loader_rebuilds_hard_positive_texts(
    generated: tuple[build.Workspace, SourceRepo, list[build.GeneratedCase]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, repo, cases = generated
    kept = cases
    for split in ("train", "dev"):
        out = ws.root / split
        build.write_private_json(out / "records.json", [c.record for c in kept])
        build.write_private_json(
            out / "mutations.json", {c.record["record_id"]: c.mutation for c in kept}
        )
    monkeypatch.setattr(loader, "get_repo", lambda _slug: repo)
    examples = loader._examples(ws.root, "train", "train", "failurelens-train-v03")
    assert len(examples) == len(kept)
    for example, case in zip(examples, kept, strict=True):
        name = case.mutation["function"]
        node = _function(MODULE, name)
        clean = "".join(MODULE.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
        assert example.positive.startswith(f"def {name}(")
        assert example.positive != clean


def test_v03_candidate_configs_reference_corpus_by_digest_only() -> None:
    configs = sorted(Path("training/configs").glob("stage2_v03_*.json"))
    assert len(configs) >= 4
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in configs]
    for config in loaded:
        assert config["corpus"] == "failurelens-corpus-v03"
        digest = config["dataset_sha256"]
        assert digest == loader.PLACEHOLDER_DIGEST or len(digest) == 64
        assert "/" not in json.dumps(config).replace("nomic-ai/CodeRankEmbed", "")
        assert config["model_revision"] == "3c4b60807d71f79b43f3c4363786d9493691f8b1"
        assert config["micro_batch_size"] == 1
    keys = ("learning_rate", "epochs", "batch_size", "temperature", "trainable_encoder_layers")
    assert len({tuple(c.get(k, 1) for k in keys) for c in loaded}) == len(loaded)
    assert any(c.get("trainable_encoder_layers", 1) == 2 for c in loaded)


class _Param:
    def __init__(self) -> None:
        self.requires_grad = True

    def requires_grad_(self, flag: bool) -> None:
        self.requires_grad = flag

    def numel(self) -> int:
        return 10


class _Module:
    def __init__(self, params: list[_Param]) -> None:
        self._params = params

    def parameters(self) -> list[_Param]:
        return self._params


class _Encoder:
    def __init__(self, layers: list[_Module]) -> None:
        self.layers = layers


class _Inner(_Module):
    def __init__(self, layers: list[_Module], extra: list[_Param]) -> None:
        super().__init__([p for layer in layers for p in layer.parameters()] + extra)
        self.encoder = _Encoder(layers)


class _Wrapper:
    def __init__(self, inner: _Inner) -> None:
        self.model = inner


class _Model:
    def __init__(self, layers: int) -> None:
        self.layer_modules = [_Module([_Param(), _Param()]) for _ in range(layers)]
        self.extra = [_Param()]
        self.inner = _Inner(self.layer_modules, self.extra)
        self.wrapper = _Wrapper(self.inner)

    def __getitem__(self, index: int) -> _Wrapper:
        assert index == 0
        return self.wrapper

    def parameters(self) -> list[_Param]:
        return self.inner.parameters()


def test_freeze_unfreezes_requested_number_of_last_layers() -> None:
    model = _Model(4)
    trainable, total = _freeze_all_but_last_encoder_layer(model)
    assert (trainable, total) == (20, 90)
    assert [all(p.requires_grad for p in m.parameters()) for m in model.layer_modules] == [
        False,
        False,
        False,
        True,
    ]
    model = _Model(4)
    trainable, _ = _freeze_all_but_last_encoder_layer(model, 2)
    assert trainable == 40
    assert not model.extra[0].requires_grad
    with pytest.raises(ValueError, match="exceeds"):
        _freeze_all_but_last_encoder_layer(_Model(2), 3)
    with pytest.raises(ValueError, match=">= 1"):
        _freeze_all_but_last_encoder_layer(_Model(2), 0)
