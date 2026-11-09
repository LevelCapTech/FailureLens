"""Private "Gate C v2" evaluation-set builder.

Builds a fresh, private, repository-disjoint evaluation set for the *final*
independent measurement of a selected candidate. It reuses the sandboxed v0.2
corpus machinery (:mod:`training.corpus.build`, :mod:`training.corpus.runner`) and
never executes third-party test code outside the network-less container (pass
``--python training/corpus/sandbox/sandbox_python.sh``).

Differences from the v0.2 dev/train generator:

* Its own repository table (role ``eval``), checked to be disjoint from every
  v0.2 train/dev repository, the legacy training repositories, the first Gate C
  hold-out and a caller-supplied list of additional training repositories.
* Extra statement-level mutation kinds, including a two-line edit, that keep the
  net line change in ``{0, +1}`` (required by the evaluator's span check).
* Failure logs carry repository-relative paths that exist in the failing tree
  (the sanitizer strips the private work-copy prefix), so the shipped retrieval
  can anchor on real files.
* Each accepted case is tagged with a structural *surfacing category* computed
  from the stack trace alone (is the root-cause function the innermost frame,
  deeper in the stack, or absent from it?). Selection is stratified on that
  category and on mutation kind with a fixed seed. No model output is consulted.

Stages (each resumable)::

    python -m training.corpus.gatec_v2 --root DIR --python SANDBOX prepare
    python -m training.corpus.gatec_v2 --root DIR --python SANDBOX generate
    python -m training.corpus.gatec_v2 --root DIR --python SANDBOX finalize
    python -m training.corpus.gatec_v2 --root DIR --python SANDBOX manifest --train-repos-file F
    python -m training.corpus.gatec_v2 --root DIR --python SANDBOX check

``manifest`` rewrites only ``train_manifest.json`` so it can be regenerated when
the training-repository list changes; it fails closed on any repository overlap.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.corpus import build
from training.corpus.build import (
    GeneratedCase,
    Workspace,
    choose_negatives,
    copy_tracked_files,
    sha256_file,
    write_private_json,
)
from training.corpus.mutations import (
    FunctionInfo,
    Mutation,
    enumerate_functions,
    enumerate_mutations,
)
from training.corpus.repos import HOLDOUT_REPOSITORIES, REPOSITORIES, SourceRepo
from training.corpus.runner import (
    PytestRun,
    extract_failure_log,
    run_pytest,
    terminal_exception,
)

__all__ = [
    "EVAL_DATASET_ID",
    "EVAL_REPOSITORIES",
    "EvalCase",
    "apply_edit",
    "classify_surfacing",
    "enumerate_extra_mutations",
    "select_cases",
    "validate_eval_table",
]

EVAL_DATASET_ID = "failurelens-gatec-v2"
EVAL_ID = "failurelens-gatec-v2"
EVAL_GENERATOR_VERSION = "gatec-v2-1.0"
EVAL_APPROVAL_ID = "owner-approved-v03-gatec-v2-2026-11-07"
DEFAULT_SEED = 20261108
#: Training repositories of the earlier private datasets (kept train-only).
LEGACY_TRAIN_REPOSITORIES: tuple[str, ...] = ("pallets/click", "python-attrs/attrs")
Log = Callable[[str], None]

KIND_SWAP = "adjacent-statement swap"
KIND_DUP = "duplicated statement"
KIND_PASS = "statement removal"
KIND_ARGSWAP = "call-argument swap"

EVAL_REPOSITORIES: tuple[SourceRepo, ...] = (
    SourceRepo(
        "tkem/cachetools",
        "cachetools",
        "1ce79efa552d1bdda70d144d401ef9105d999184",
        "MIT",
        "eval",
        ("src/cachetools/",),
        (),
        ("tests",),
        "src",
        "LICENSE",
    ),
    SourceRepo(
        "PyCQA/pycodestyle",
        "pycodestyle",
        "10a4427c75740717b43448339fcf71f11bc33d1a",
        "MIT",
        "eval",
        ("pycodestyle.py",),
        (),
        ("testsuite",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "python-validators/validators",
        "validators",
        "e2e49fb3bb4b9f1ce955638386aa167356d24055",
        "MIT",
        "eval",
        ("src/validators/",),
        (),
        ("tests", "--ignore=tests/crypto_addresses"),
        "src",
        "LICENSE.txt",
    ),
    SourceRepo(
        "bottlepy/bottle",
        "bottle",
        "b0a0a1014f7a7e6f6b7c193886a0913f46ee3ee6",
        "MIT",
        "eval",
        ("bottle.py",),
        (),
        (
            "test",
            "--ignore=test/test_jinja2.py",
            "--ignore=test/test_mako.py",
            "--ignore=test/test_importhook.py",
        ),
        ".",
        "LICENSE",
    ),
)


# ----------------------------------------------------------------------- table checks


def validate_eval_table(
    repos: tuple[SourceRepo, ...] = EVAL_REPOSITORIES,
    extra_train: tuple[str, ...] = (),
) -> None:
    """Fail closed on any overlap with train/dev/hold-out repositories or bad pins."""
    slugs = [r.slug for r in repos]
    keys = [r.key for r in repos]
    if len(set(slugs)) != len(slugs) or len(set(keys)) != len(keys):
        raise ValueError("duplicate repository entry")
    used = {r.slug.lower() for r in REPOSITORIES} | {s.lower() for s in LEGACY_TRAIN_REPOSITORIES}
    used |= {s.lower() for s in HOLDOUT_REPOSITORIES} | {s.lower() for s in extra_train}
    used_keys = {r.key for r in REPOSITORIES}
    for repo in repos:
        if repo.slug.lower() in used or repo.key in used_keys:
            raise ValueError(f"repository overlaps train, dev or hold-out sources: {repo.key}")
        if repo.role != "eval":
            raise ValueError(f"invalid role for {repo.key}")
        if len(repo.commit) != 40 or any(c not in "0123456789abcdef" for c in repo.commit):
            raise ValueError(f"invalid pin for {repo.key}")
        if repo.license not in {"MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0"}:
            raise ValueError(f"license not allowed for {repo.key}")


# ------------------------------------------------------------------ general mutation edit


def apply_edit(source: str, mutation: Mutation) -> str:
    """Apply ``mutation`` whose ``old`` may span several consecutive lines."""
    lines = source.split("\n")
    old_lines = mutation.old.split("\n")
    start = mutation.line_no - 1
    if lines[start : start + len(old_lines)] != old_lines:
        raise ValueError("mutation does not match the source lines")
    lines[start : start + len(old_lines)] = mutation.new.split("\n")
    return "\n".join(lines)


def _simple_stmt(node: ast.stmt) -> bool:
    if node.end_lineno != node.lineno:
        return False
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        return True
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)


def _names(node: ast.AST, ctx: type) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


def _stmt_lists(func: ast.FunctionDef) -> Iterator[list[ast.stmt]]:
    for node in ast.walk(func):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def _docstring_end(func: ast.FunctionDef) -> int:
    first = func.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.end_lineno or first.lineno
    return 0


def _arg_swap(line: str, call: ast.Call) -> str | None:
    """Swap the first two plain positional arguments of a one-line call."""
    if len(call.args) < 2 or any(isinstance(a, ast.Starred) for a in call.args[:2]):
        return None
    first, second = call.args[0], call.args[1]
    if (
        first.lineno != second.lineno
        or first.end_lineno != first.lineno
        or second.end_lineno != second.lineno
        or first.end_col_offset is None
        or second.end_col_offset is None
    ):
        return None
    raw = line.encode("utf-8")
    a = raw[first.col_offset : first.end_col_offset]
    b = raw[second.col_offset : second.end_col_offset]
    if a == b:
        return None
    swapped = (
        raw[: first.col_offset]
        + b
        + raw[first.end_col_offset : second.col_offset]
        + a
        + raw[second.end_col_offset :]
    )
    return swapped.decode("utf-8")


def enumerate_extra_mutations(source: str, info: FunctionInfo) -> list[Mutation]:
    """Statement-level edits: adjacent swap (two lines), duplication, removal, argument swap.

    Every result keeps the net line change at 0 or +1, parses, differs from the
    source, and its ``old`` text occurs exactly once in the function text.
    """
    tree = ast.parse(source)
    func = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == info.name and n.lineno == info.start_line
    )
    lines = source.splitlines()
    function_text = "\n".join(lines[info.start_line - 1 : info.end_line])
    doc_end = _docstring_end(func)
    found: list[Mutation] = []
    seen: set[tuple[int, str]] = set()

    def accept(mutation: Mutation) -> None:
        if mutation.old == mutation.new or function_text.count(mutation.old) != 1:
            return
        key = (mutation.line_no, mutation.new)
        if key in seen:
            return
        try:
            ast.parse(apply_edit(source, mutation) + ("" if source.endswith("\n") else "\n"))
        except (SyntaxError, ValueError):
            return
        seen.add(key)
        found.append(mutation)

    for block in _stmt_lists(func):
        for index, stmt in enumerate(block):
            if stmt.lineno <= doc_end or not _simple_stmt(stmt):
                continue
            text = lines[stmt.lineno - 1]
            indent = text[: len(text) - len(text.lstrip())]
            if isinstance(stmt, (ast.AugAssign, ast.Expr)):
                accept(Mutation(info.name, KIND_DUP, stmt.lineno, text, f"{text}\n{text}", False))
            accept(Mutation(info.name, KIND_PASS, stmt.lineno, text, f"{indent}pass", True))
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    swapped = _arg_swap(text, node)
                    if swapped is not None:
                        accept(Mutation(info.name, KIND_ARGSWAP, stmt.lineno, text, swapped, True))
            if index + 1 < len(block):
                nxt = block[index + 1]
                if (
                    _simple_stmt(nxt)
                    and nxt.lineno == stmt.end_lineno + 1  # type: ignore[operator]
                    and nxt.lineno > doc_end
                ):
                    a_defs = _names(stmt, ast.Store)
                    b_defs = _names(nxt, ast.Store)
                    dependent = (a_defs & _names(nxt, ast.Load)) or (
                        b_defs & _names(stmt, ast.Load)
                    )
                    other = lines[nxt.lineno - 1]
                    if dependent and other != text:
                        accept(
                            Mutation(
                                info.name,
                                KIND_SWAP,
                                stmt.lineno,
                                f"{text}\n{other}",
                                f"{other}\n{text}",
                                True,
                            )
                        )
    return found


# ------------------------------------------------------------------- surfacing category

_FRAME = re.compile(r'^\s*File "([^"]+)", line \d+, in (\S+)', re.MULTILINE)
_TEST_PATH = re.compile(r"(^|/)(tests?|testsuite|testing)/|(^|/)test_[^/]*\.py$|conftest\.py$")

CATEGORY_INNERMOST = "root-is-innermost-frame"
CATEGORY_MID = "root-deeper-in-stack"
CATEGORY_ABSENT = "root-absent-from-stack"
CATEGORIES = (CATEGORY_INNERMOST, CATEGORY_MID, CATEGORY_ABSENT)


def classify_surfacing(stack_trace: str, file: str, function: str) -> str:
    """Where does the failure surface relative to the root-cause function?

    Uses only the stack trace text. Frames in test files are ignored when finding
    the innermost production frame. ``root-is-innermost-frame``: the last non-test
    frame is the changed function. ``root-deeper-in-stack``: the changed function
    appears, but the failure surfaces in a frame called from it or in a test frame
    after it. ``root-absent-from-stack``: the changed function is not in the stack.
    """
    frames = [(p, name) for p, name in _FRAME.findall(stack_trace)]
    if (file, function) not in frames:
        return CATEGORY_ABSENT
    production = [(p, n) for p, n in frames if not _TEST_PATH.search(p)]
    if production and production[-1] == (file, function) and frames[-1] == (file, function):
        return CATEGORY_INNERMOST
    return CATEGORY_MID


# --------------------------------------------------------------------------- generation


@dataclass
class EvalCase:
    record: dict[str, Any]
    mutation: dict[str, Any]
    category: str
    kind: str
    repo: str
    function_key: str


def _mutation_dict(file: str, mutation: Mutation) -> dict[str, Any]:
    return build._mutation_dict(file, mutation)


def evaluate_edit(
    ws: Workspace,
    repo: SourceRepo,
    file: str,
    mutation: Mutation,
    source: str,
    candidate_tests: list[str],
    passed: set[str],
) -> str | tuple[str, PytestRun, PytestRun, tuple[str, str]]:
    """Same pass-before/fail-after protocol as :func:`build.evaluate_mutation`, multi-line safe."""
    work = ws.work(repo)
    path = work / file
    original = path.read_bytes()
    tests = sorted(set(candidate_tests) & passed)[: build.MAX_TESTS_PER_MUTATION]
    skip = tuple({*repo.skip_prefixes, *(f"{t}/" for t in repo.test_args if not t.startswith("-"))})
    try:
        path.write_text(apply_edit(source, mutation), encoding="utf-8")
        first = run_pytest(
            ws.python,
            work,
            ["-x", *tests],
            pythonpath=repo.pythonpath,
            skip_prefixes=skip,
            trace_calls=False,
            timeout=300,
        )
        failed = [n for n, o in first.outcomes.items() if o == "failed"]
        if first.timed_out or not failed:
            return "not_detected_by_tests"
        node = failed[0]
        second = run_pytest(
            ws.python,
            work,
            [node],
            pythonpath=repo.pythonpath,
            skip_prefixes=skip,
            trace_calls=False,
            timeout=120,
        )
    finally:
        path.write_bytes(original)
    if second.returncode != 1 or second.outcomes.get(node) != "failed":
        return "rejected_flaky"
    log_text = extract_failure_log(second.stdout, work)
    if log_text is None:
        return "rejected_log"
    pre = run_pytest(
        ws.python,
        work,
        [node],
        pythonpath=repo.pythonpath,
        skip_prefixes=skip,
        trace_calls=False,
        timeout=120,
    )
    if pre.returncode != 0 or pre.outcomes.get(node) != "passed":
        return "rejected_flaky"
    return node, second, pre, log_text


def _baseline(ws: Workspace, repo: SourceRepo) -> PytestRun:
    skip = tuple(
        {
            *repo.skip_prefixes,
            *(f"{t}/" for t in repo.test_args if not t.startswith("-")),
            "docs/",
        }
    )
    return run_pytest(
        ws.python,
        ws.work(repo),
        list(repo.test_args),
        pythonpath=repo.pythonpath,
        skip_prefixes=skip,
        timeout=1800,
    )


def generate_eval_repo(
    ws: Workspace,
    repo: SourceRepo,
    *,
    seed: int,
    target: int,
    per_function: int = 3,
    max_attempts: int | None = None,
    log: Log = build._log,
    hard: bool = False,
    dataset_id: str = EVAL_DATASET_ID,
    generator_version: str = EVAL_GENERATOR_VERSION,
    categorize: Callable[[str, str, str, Path], str] | None = None,
) -> tuple[list[EvalCase], dict[str, Any]]:
    """Generate up to ``target`` execution-validated candidate cases (oversupply for selection).

    ``hard`` also enumerates the v0.3 hard text shapes; ``categorize(stack_trace, file,
    function, work_tree)`` replaces the default surfacing classifier. The defaults
    reproduce the Gate C v2 behaviour exactly.
    """
    work = ws.work(repo)
    started = time.monotonic()
    base = _baseline(ws, repo)
    passed = {n for n, o in base.outcomes.items() if o == "passed"}
    report: dict[str, Any] = {
        "repo": repo.slug,
        "baseline_passed_tests": len(passed),
        "baseline_other_outcomes": len(base.outcomes) - len(passed),
        "attempted": 0,
        "accepted": 0,
        "not_detected_by_tests": 0,
        "rejected_log": 0,
        "rejected_flaky": 0,
    }
    if len(passed) < 10:
        report["note"] = "too few passing baseline tests"
        return [], report
    callers: dict[tuple[str, str], list[str]] = {}
    for node, calls in base.calls.items():
        if node in passed:
            for file, name, _ in calls:
                callers.setdefault((file, name), []).append(node)
    sources: dict[str, str] = {}
    infos: dict[str, list[FunctionInfo]] = {}

    def functions_of(file: str) -> list[FunctionInfo]:
        if file not in infos:
            path = work / file
            text = path.read_text(encoding="utf-8") if path.is_file() else ""
            sources[file] = text
            infos[file] = enumerate_functions(text)
        return infos[file]

    rng = random.Random(f"{seed}:{repo.key}")
    plan: list[tuple[str, FunctionInfo]] = []
    for file in build.source_files(repo, work):
        try:
            for info in functions_of(file):
                if (file, info.name) in callers:
                    plan.append((file, info))
        except UnicodeDecodeError:
            continue
    rng.shuffle(plan)
    attempts_cap = max_attempts or target * 10
    cases: list[EvalCase] = []
    kinds: Counter[str] = Counter()
    for file, info in plan:
        if len(cases) >= target or report["attempted"] >= attempts_cap:
            break
        mutations = enumerate_mutations(sources[file], info, hard=hard) + enumerate_extra_mutations(
            sources[file], info
        )
        rng.shuffle(mutations)
        mutations.sort(key=lambda m: kinds[m.kind])
        used: set[str] = set()
        for mutation in mutations:
            if len(used) >= per_function or len(cases) >= target:
                break
            if mutation.kind in used or report["attempted"] >= attempts_cap:
                continue
            report["attempted"] += 1
            outcome = evaluate_edit(
                ws, repo, file, mutation, sources[file], callers[(file, info.name)], passed
            )
            if isinstance(outcome, str):
                report[outcome] += 1
                continue
            node, post, pre, log_text = outcome
            called = {(f, n) for f, n, _ in base.calls.get(node, [])}
            negatives = choose_negatives(file, info.name, called, functions_of)
            if not negatives:
                report["rejected_log"] += 1
                continue
            exception = terminal_exception(log_text[1])
            record = build.build_record(
                repo,
                len(cases),
                file,
                info,
                mutation,
                node,
                pre,
                post,
                log_text,
                negatives,
                exception,
            )
            record["dataset_id"] = dataset_id
            record["generator_version"] = generator_version
            category = (
                categorize(log_text[1], file, info.name, work)
                if categorize is not None
                else classify_surfacing(log_text[1], file, info.name)
            )
            cases.append(
                EvalCase(
                    record,
                    _mutation_dict(file, mutation),
                    category,
                    mutation.kind,
                    repo.slug,
                    f"{file}::{info.name}",
                )
            )
            used.add(mutation.kind)
            kinds[mutation.kind] += 1
        if cases and len(cases) % 10 == 0:
            log(f"{repo.key}: {len(cases)}/{target} (attempted {report['attempted']})")
    report["accepted"] = len(cases)
    report["by_kind"] = dict(kinds)
    report["by_category"] = dict(Counter(c.category for c in cases))
    report["seconds"] = round(time.monotonic() - started, 1)
    return cases, report


# --------------------------------------------------------------------------- selection

#: Desired share of each surfacing category in the final set (structural, not score based).
CATEGORY_SHARE: dict[str, float] = {
    CATEGORY_INNERMOST: 0.30,
    CATEGORY_MID: 0.45,
    CATEGORY_ABSENT: 0.25,
}


def select_cases(
    cases: list[EvalCase],
    total: int,
    *,
    seed: int,
    per_function: int = 2,
    shares: dict[str, float] | None = None,
) -> list[EvalCase]:
    """Seeded, stratified selection by repository, surfacing category and mutation kind.

    Uses structural metadata only. At most ``per_function`` cases per function; the
    repository quota is balanced, with any shortfall redistributed to other
    repositories. Deterministic for a given input and seed.
    """
    rng = random.Random(f"select:{seed}")
    by_repo: dict[str, list[EvalCase]] = {}
    for case in sorted(cases, key=lambda c: (c.repo, c.record["record_id"])):
        by_repo.setdefault(case.repo, []).append(case)
    for pool in by_repo.values():
        rng.shuffle(pool)
    repos = sorted(by_repo)
    quota = dict.fromkeys(repos, total // len(repos)) if repos else {}
    for repo in repos[: total - sum(quota.values())]:
        quota[repo] += 1
    wanted = {c: max(1, round(total * s)) for c, s in (shares or CATEGORY_SHARE).items()}
    picked: list[EvalCase] = []
    chosen: set[int] = set()
    function_count: Counter[str] = Counter()
    kind_count: Counter[str] = Counter()
    category_count: Counter[str] = Counter()

    def take(case: EvalCase) -> None:
        picked.append(case)
        chosen.add(id(case))
        function_count[case.function_key] += 1
        kind_count[case.kind] += 1
        category_count[case.category] += 1

    def next_case(pool: list[EvalCase]) -> EvalCase | None:
        ranked = [
            c for c in pool if id(c) not in chosen and function_count[c.function_key] < per_function
        ]
        if not ranked:
            return None
        # prefer the category furthest below its target share, then the rarest kind
        ranked.sort(
            key=lambda c: (
                category_count[c.category] - wanted.get(c.category, 0),
                kind_count[c.kind],
                function_count[c.function_key],
            )
        )
        return ranked[0]

    remaining = dict(quota)
    for _ in range(total * 2):
        progressed = False
        for repo in repos:
            if remaining[repo] <= 0 or len(picked) >= total:
                continue
            case = next_case(by_repo[repo])
            if case is None:
                remaining[repo] = 0
                continue
            take(case)
            remaining[repo] -= 1
            progressed = True
        if len(picked) >= total:
            break
        if not progressed:
            # redistribute shortfall to repositories that still have candidates
            spare = [r for r in repos if next_case(by_repo[r]) is not None]
            if not spare:
                break
            for repo in spare:
                remaining[repo] = remaining.get(repo, 0) + 1
    return sorted(picked, key=lambda c: (c.repo, c.function_key, c.kind))


# --------------------------------------------------------------------------- finalize


def _renumber(cases: list[EvalCase]) -> None:
    counters: Counter[str] = Counter()
    for case in cases:
        key = case.repo.split("/", 1)[1].lower().replace(".", "-")
        case.record["record_id"] = f"gcv2-{key}-{counters[key]:04d}"
        counters[key] += 1


def write_eval_trees(
    ws: Workspace, cases: list[EvalCase], out_dir: Path, resolve: Callable[[str], SourceRepo]
) -> None:
    """One failing-state tree per case plus the evaluator's source and tree manifests."""
    from failurelens.config import IndexConfig
    from failurelens.indexing.repository import iter_source_files

    snapshots: dict[str, Any] = {}
    entries: dict[str, Any] = {}
    repositories: dict[str, Any] = {}
    for case in cases:
        record, mutation = case.record, case.mutation
        repo = resolve(record["source_repository"])
        clone = ws.source(repo)
        root = ws.root / "trees" / record["record_id"]
        copy_tracked_files(clone, root)
        target = root / mutation["file"]
        edited = apply_edit(
            target.read_text(encoding="utf-8"),
            Mutation(**{k: mutation[k] for k in Mutation.__dataclass_fields__}),
        )
        target.write_text(edited, encoding="utf-8")
        indexed = {
            relative: sha256_file(path) for path, relative in iter_source_files(root, IndexConfig())
        }
        entries[record["record_id"]] = {
            "base_repository": str(clone),
            "root": str(root),
            "commit": record["source_commit"],
            "indexed_files": indexed,
            "mutation": {k: mutation[k] for k in ("file", "function", "old", "new")},
        }
        files = {item["location"]["file"] for item in record["candidates"]}
        snapshots[record["record_id"]] = {
            f: {"path": str(root / f), "sha256": sha256_file(root / f)} for f in sorted(files)
        }
        repositories[record["source_repository"]] = {
            "commit": record["source_commit"],
            "rights_cleared": True,
            "license": record["source_license"],
        }
    write_private_json(
        out_dir / "source_manifest.json", {"repositories": repositories, "snapshots": snapshots}
    )
    write_private_json(out_dir / "tree_manifest.json", {"cases": entries})


def default_train_repositories() -> list[str]:
    """All repositories used to train or tune any earlier candidate (train and dev)."""
    return sorted({r.slug for r in REPOSITORIES} | set(LEGACY_TRAIN_REPOSITORIES))


def write_train_manifest(out_dir: Path, train_repos: list[str], eval_repos: list[str]) -> Path:
    """(Re)write the approved-training-repository manifest; reject any overlap."""
    overlap = {r.lower() for r in train_repos} & {r.lower() for r in eval_repos}
    if overlap or {r.lower() for r in eval_repos} & {r.lower() for r in HOLDOUT_REPOSITORIES}:
        raise RuntimeError("repository overlap between training and evaluation sources")
    path = out_dir / "train_manifest.json"
    write_private_json(
        path,
        {"approval_id": EVAL_APPROVAL_ID, "training_repositories": sorted(set(train_repos))},
    )
    return path


def finalize(
    ws: Workspace,
    *,
    total: int,
    seed: int,
    train_repos: list[str],
    log: Log = build._log,
) -> dict[str, Any]:
    validate_eval_table(extra_train=tuple(train_repos))
    pool: list[EvalCase] = []
    repo_reports: list[dict[str, Any]] = []
    for repo in EVAL_REPOSITORIES:
        raw_path = ws.raw(repo)
        if not raw_path.exists():
            log(f"skip {repo.key}: no raw output")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        repo_reports.append(raw["report"])
        pool.extend(
            EvalCase(c["record"], c["mutation"], c["category"], c["kind"], repo.slug, c["function"])
            for c in raw["cases"]
        )
    wrapped = [GeneratedCase(c.record, c.mutation) for c in pool]
    kept_generated, validation = build.validate_cases(wrapped)
    kept_ids = {id(g) for g in kept_generated}
    survivors = [c for c, g in zip(pool, wrapped, strict=True) if id(g) in kept_ids]
    selected = select_cases(survivors, total, seed=seed)
    _renumber(selected)
    out_dir = ws.root
    records = [c.record for c in selected]
    write_private_json(out_dir / "records.json", records)
    write_private_json(
        out_dir / "case_meta_private.json",
        {
            c.record["record_id"]: {
                "category": c.category,
                "kind": c.kind,
                "function": c.function_key,
            }
            for c in selected
        },
    )
    by_slug = {r.slug: r for r in EVAL_REPOSITORIES}
    write_eval_trees(ws, selected, out_dir, lambda slug: by_slug[slug])
    eval_repos = sorted({c.repo for c in selected})
    write_train_manifest(out_dir, train_repos, eval_repos)
    digest = sha256_file(out_dir / "records.json")
    summary: dict[str, Any] = {
        "eval_id": EVAL_ID,
        "generator_version": EVAL_GENERATOR_VERSION,
        "records": len(records),
        "repositories": eval_repos,
        "records_by_repository": dict(Counter(c.repo for c in selected)),
        "by_mutation": dict(Counter(c.kind for c in selected)),
        "by_surfacing_category": dict(Counter(c.category for c in selected)),
        "candidate_pool_size": len(pool),
        "validation": validation,
        "multi_line_edits": sum("\n" in c.mutation["old"] for c in selected),
        "net_plus_one_edits": sum(c.mutation["in_place"] is False for c in selected),
        "records_sha256": digest,
        "train_manifest_repositories": len(train_repos),
        "repositories_report": repo_reports,
    }
    write_private_json(
        out_dir / "manifest.json",
        {
            "eval_id": EVAL_ID,
            "generator_version": EVAL_GENERATOR_VERSION,
            "purpose": "private final independent evaluation; one use per selected candidate",
            "records_sha256": digest,
            "pins": {r.slug: r.commit for r in EVAL_REPOSITORIES if r.slug in set(eval_repos)},
            "excluded_repositories": sorted(
                {*HOLDOUT_REPOSITORIES, *(r.slug for r in REPOSITORIES), *LEGACY_TRAIN_REPOSITORIES}
            ),
        },
    )
    write_private_json(out_dir / "report.json", summary)
    return summary


# --------------------------------------------------------------------------- check


def check(
    ws: Workspace, *, top_k: int = 5, categories: tuple[str, ...] = CATEGORIES
) -> dict[str, Any]:
    """Structural sanity only: load through the product pool, report pool recall.

    No model is loaded and no scores are produced. Reports retrieval abstentions and
    how often the shipped retrieval pool contains the labelled root-cause function.
    """
    from evaluation.product_pool import load_product_cases

    root = ws.root
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "case_meta_private.json").read_text(encoding="utf-8"))
    digest = sha256_file(root / "records.json")
    cases, _ = load_product_cases(
        root / "records.json",
        digest,
        root / "source_manifest.json",
        root / "train_manifest.json",
        root / "tree_manifest.json",
        top_k=top_k,
        production=True,
        allow_ambiguous_retrieved_names=True,
    )
    by_category: dict[str, list[int]] = {c: [0, 0, 0] for c in categories}  # n, recalled, empty
    total_recalled = abstained = 0
    for record, case in zip(records, cases, strict=True):
        # compare function identities (file::name): candidate ids embed line spans, which
        # legitimately shift by one line in the failing tree for inserted statements
        functions = {c.function_id for c in case.request.candidates}
        recalled = bool(case.positive_function_ids and functions & case.positive_function_ids)
        empty = not case.request.candidates
        row = by_category[meta[record["record_id"]]["category"]]
        row[0] += 1
        row[1] += int(recalled)
        row[2] += int(empty)
        total_recalled += int(recalled)
        abstained += int(empty)
    n = len(cases)
    return {
        "cases": n,
        "repositories": len({r["source_repository"] for r in records}),
        "retrieval_abstentions": abstained,
        "positive_function_in_pool": total_recalled,
        "pool_function_recall": round(total_recalled / n, 4) if n else None,
        "by_category": {
            c: {"cases": v[0], "pool_recalled": v[1], "abstained": v[2]}
            for c, v in by_category.items()
        },
        "note": "structural pool check only; no model run, no ranking scores",
    }


# ------------------------------------------------------------------------------ CLI


def _eval_repo(key: str) -> SourceRepo:
    for repo in EVAL_REPOSITORIES:
        if repo.key == key or repo.slug == key:
            return repo
    raise KeyError(key)


def _read_train_repos(path: Path | None) -> list[str]:
    if path is None:
        return default_train_repositories()
    value = json.loads(path.read_text(encoding="utf-8"))
    repos = value["training_repositories"] if isinstance(value, dict) else value
    if not isinstance(repos, list) or any(not isinstance(r, str) or not r.strip() for r in repos):
        raise ValueError("train repositories must be a list of 'owner/name' strings")
    return sorted({*default_train_repositories(), *repos})


def _generate_one(args: argparse.Namespace, ws: Workspace, repo: SourceRepo) -> None:
    raw_path = ws.raw(repo)
    if raw_path.exists() and not args.force:
        build._log(f"skip {repo.key}: raw output exists")
        return
    cases, report = generate_eval_repo(ws, repo, seed=args.seed, target=args.repo_target)
    write_private_json(
        raw_path,
        {
            "report": report,
            "cases": [
                {
                    "record": c.record,
                    "mutation": c.mutation,
                    "category": c.category,
                    "kind": c.kind,
                    "function": c.function_key,
                }
                for c in cases
            ],
        },
    )
    summary = {k: v for k, v in report.items() if k not in {"by_kind", "by_category"}}
    build._log(f"done {repo.key}: {json.dumps(summary)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["prepare", "generate", "finalize", "manifest", "check"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="sandboxed python launcher")
    parser.add_argument("--repos", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repo-target", type=int, default=24, help="oversupply per repository")
    parser.add_argument("--total", type=int, default=40, help="final number of cases")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--train-repos-file",
        type=Path,
        default=None,
        help="JSON list (or {training_repositories: [...]}) of extra training repositories "
        "to add to the v0.2 train/dev/legacy set in train_manifest.json",
    )
    args = parser.parse_args(argv)
    ws = Workspace(args.root.expanduser(), args.python.expanduser())
    ws.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    ws.root.chmod(0o700)
    train_repos = _read_train_repos(args.train_repos_file)
    validate_eval_table(extra_train=tuple(train_repos))
    repos = [r for r in EVAL_REPOSITORIES if args.repos is None or r.key in args.repos]
    if args.stage == "prepare":
        evidence = {r.slug: build.prepare_repo(ws, r) for r in repos}
        write_private_json(ws.root / "license_evidence.json", evidence)
        print(json.dumps(evidence, indent=2))
    elif args.stage == "generate":
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda r: _generate_one(args, ws, r), repos))
    elif args.stage == "finalize":
        print(
            json.dumps(
                finalize(ws, total=args.total, seed=args.seed, train_repos=train_repos), indent=2
            )
        )
    elif args.stage == "manifest":
        meta = json.loads((ws.root / "manifest.json").read_text(encoding="utf-8"))
        eval_repos = sorted(meta["pins"])
        path = write_train_manifest(ws.root, train_repos, eval_repos)
        print(json.dumps({"train_manifest": path.name, "training_repositories": len(train_repos)}))
    else:
        print(json.dumps(check(ws), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
