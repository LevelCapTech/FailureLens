"""Private v0.2 fault-corpus builder (train split and dev split).

Generates pass-before/fail-after records by applying one exact single-line
mutation (:mod:`training.corpus.mutations`) to a module-level function of a
pinned, license-reviewed repository and running that project's own pytest tests
(:mod:`training.corpus.runner`). The generator is opt-in, executes third-party
test code, and is never invoked by CI. All outputs go to a private directory
(mode 0700, files 0600) outside Git; only aggregate reports may be quoted.

Stages (each resumable)::

    python -m training.corpus.build --root DIR --python VENV_PY prepare
    python -m training.corpus.build --root DIR --python VENV_PY generate
    python -m training.corpus.build --root DIR --python VENV_PY finalize

``prepare`` clones the pinned repositories, verifies pins and license text, and
creates clean work copies. ``generate`` writes one raw file per repository.
``finalize`` validates (schema + semantic gates + duplicates), writes the
train/dev record sets, dev failing trees, evaluator manifests and an aggregate
report, and prints the corpus digest used by training configs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from failurelens.quality import validate_record_set
from training.corpus.mutations import (
    HARD_KINDS,
    KIND_CALLEE,
    FunctionInfo,
    Mutation,
    apply_mutation,
    enumerate_functions,
    enumerate_mutations,
)
from training.corpus.repos import (
    HOLDOUT_REPOSITORIES,
    REPOSITORIES,
    SourceRepo,
    get_repo,
    validate_table,
)
from training.corpus.runner import PytestRun, extract_failure_log, run_pytest, terminal_exception


@dataclass(frozen=True)
class Profile:
    """Identifiers and generation defaults of one corpus version."""

    name: str
    train_dataset_id: str
    dev_dataset_id: str
    corpus_id: str
    generator_version: str
    approval_id: str
    hard: bool  # also enumerate the v0.3 hard mutation kinds
    per_function: int
    seed: int


PROFILES: dict[str, Profile] = {
    "v02": Profile(
        "v02",
        "failurelens-train-v02",
        "failurelens-dev-v02",
        "failurelens-corpus-v02",
        "corpus-v02-1.0",
        "owner-delegated-v02-cycle-2026-11-07",
        False,
        2,
        20261107,
    ),
    "v03": Profile(
        "v03",
        "failurelens-train-v03",
        "failurelens-dev-v03",
        "failurelens-corpus-v03",
        "corpus-v03-1.0",
        "owner-delegated-v03-cycle-2026-11-07",
        True,
        3,
        20261108,
    ),
}
#: Active profile; ``main`` selects it with ``--profile`` (default keeps v0.2 behaviour).
PROFILE: Profile = PROFILES["v02"]
TRAIN_DATASET_ID = PROFILES["v02"].train_dataset_id
DEV_DATASET_ID = PROFILES["v02"].dev_dataset_id
CORPUS_ID = PROFILES["v02"].corpus_id
GENERATOR_VERSION = PROFILES["v02"].generator_version
APPROVAL_ID = PROFILES["v02"].approval_id
#: Training repositories of earlier private datasets that remain train-only.
LEGACY_TRAIN_REPOSITORIES: tuple[str, ...] = ("pallets/click", "python-attrs/attrs")


def set_profile(name: str) -> Profile:
    """Select the corpus version the generator and finalizer operate on."""
    global PROFILE
    PROFILE = PROFILES[name]
    return PROFILE


def active_repositories() -> tuple[SourceRepo, ...]:
    """Repositories that belong to the active profile."""
    return tuple(r for r in REPOSITORIES if PROFILE.name in r.profiles)


MAX_FILE_BYTES = 300_000
MAX_TESTS_PER_MUTATION = 25
Log = Callable[[str], None]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_private_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check, timeout=300
    )
    return done.stdout


@dataclass(frozen=True)
class Workspace:
    root: Path
    python: Path

    def source(self, repo: SourceRepo) -> Path:
        return self.root / "sources" / repo.key

    def work(self, repo: SourceRepo) -> Path:
        return self.root / "work" / repo.key

    def raw(self, repo: SourceRepo) -> Path:
        return self.root / "raw" / f"{repo.key}.json"


# --------------------------------------------------------------------------- prepare


def classify_license(text: str) -> str | None:
    """Recognize the permissive license families used by the corpus table."""
    lowered = " ".join(text.lower().split())
    if "permission is hereby granted, free of charge" in lowered:
        return "MIT"
    if "redistribution and use in source and binary forms" in lowered:
        if "neither the name" in lowered or "to endorse or promote products" in lowered:
            return "BSD-3-Clause"
        return "BSD-2-Clause"
    if "apache license" in lowered and "version 2.0" in lowered:
        return "Apache-2.0"
    return None


def license_evidence(clone: Path, repo: SourceRepo) -> dict[str, str]:
    """Read the license file at the pin; fail unless it matches the declared SPDX id."""
    text = _git(clone, "show", f"{repo.commit}:{repo.license_file}")
    found = classify_license(text)
    if found != repo.license:
        raise ValueError(f"license text of {repo.key} is {found!r}, expected {repo.license!r}")
    return {
        "license_file": repo.license_file,
        "license_blob_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "spdx": found,
    }


def copy_tracked_files(clone: Path, dest: Path) -> None:
    """Copy exactly the files tracked at HEAD (no .git, no untracked files)."""
    names = _git(clone, "ls-files", "-z").split("\0")
    if dest.exists():
        shutil.rmtree(dest)
    for name in filter(None, names):
        source = clone / name
        if source.is_symlink() or not source.is_file():
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def prepare_repo(ws: Workspace, repo: SourceRepo, log: Log = _log) -> dict[str, str]:
    validate_table()
    clone = ws.source(repo)
    if not clone.exists():
        clone.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run(
            ["git", "clone", "-q", repo.url, str(clone)], check=True, timeout=900, text=True
        )
    if _git(clone, "rev-parse", "HEAD").strip() != repo.commit:
        _git(clone, "checkout", "-q", "--detach", repo.commit)
    if _git(clone, "status", "--porcelain").strip():
        raise RuntimeError(f"{repo.key}: pinned clone is not clean")
    evidence = license_evidence(clone, repo)
    copy_tracked_files(clone, ws.work(repo))
    log(f"prepared {repo.key} @ {repo.commit[:12]} ({evidence['spdx']})")
    return evidence


# -------------------------------------------------------------------------- generate


def source_files(repo: SourceRepo, work: Path) -> list[str]:
    result: list[str] = []
    for prefix in repo.source_prefixes:
        base = work / prefix
        candidates = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in candidates:
            rel = path.relative_to(work).as_posix()
            if rel.startswith(repo.skip_prefixes) or not rel.endswith(".py"):
                continue
            if "/tests/" in rel or "/test_" in rel or rel.rsplit("/", 1)[-1].startswith("test_"):
                continue
            if path.stat().st_size > MAX_FILE_BYTES or b"\r" in path.read_bytes():
                continue
            result.append(rel)
    return sorted(set(result))


def baseline(ws: Workspace, repo: SourceRepo, *, timeout: float = 1800.0) -> PytestRun:
    skip = tuple({*repo.skip_prefixes, *(f"{t}/" for t in repo.test_args), "docs/"})
    return run_pytest(
        ws.python,
        ws.work(repo),
        list(repo.test_args),
        pythonpath=repo.pythonpath,
        skip_prefixes=skip,
        timeout=timeout,
    )


@dataclass
class GeneratedCase:
    record: dict[str, Any]
    mutation: dict[str, Any]


def _location(file: str, info: FunctionInfo) -> dict[str, Any]:
    return {
        "file": file,
        "function": info.name,
        "start_line": info.start_line,
        "end_line": info.end_line,
    }


def _rationale(
    file: str, mutation: Mutation, exception: str, node: str, negatives: list[str]
) -> str:
    place = (
        "inserted just before the first statement" if not mutation.in_place else "edited in place"
    )
    span = (
        f"lines {mutation.line_no}-{mutation.line_no + mutation.old_line_count - 1}"
        if mutation.old_line_count > 1
        else f"line {mutation.line_no}"
    )
    return (
        f"The sole source change is a {mutation.kind} in {file}:{mutation.function} "
        f"({span}), {place}; the same test passed on the clean tree "
        f"and failed after the change. The failing test {node} only surfaces the "
        f"symptom ({exception}); the changed function is the root cause. The "
        f"negative candidate(s) {', '.join(negatives)} are unmodified functions "
        "of the same project that are not the cause of this failure."
    )


def choose_negatives(
    file: str,
    function: str,
    called: set[tuple[str, str]],
    functions_of: Callable[[str], list[FunctionInfo]],
) -> list[tuple[str, FunctionInfo, str]]:
    """Up to two unmodified negatives: one executed by the failing test, one never executed."""
    hard: list[tuple[str, FunctionInfo, str]] = []
    easy: list[tuple[str, FunctionInfo, str]] = []
    for called_file, name in sorted(called):
        if (called_file, name) == (file, function):
            continue
        match = next((f for f in functions_of(called_file) if f.name == name), None)
        if match is not None:
            hard.append((called_file, match, "executed"))
    hard.sort(key=lambda item: (item[0] != file, item[0], item[1].name))
    for info in functions_of(file):
        if info.name != function and (file, info.name) not in called:
            easy.append((file, info, "not-executed"))
    easy.sort(key=lambda item: item[1].start_line)
    picked: list[tuple[str, FunctionInfo, str]] = []
    if hard:
        picked.append(hard[0])
    if easy:
        picked.append(easy[0])
    return picked


_TRACE_FRAME = re.compile(r'File "([^"]+)", line \d+, in (\S+)')


def classify_surfacing(
    repo: SourceRepo, info: FunctionInfo, mutation: Mutation, stack_trace: str
) -> tuple[Mutation, bool]:
    """Relabel a case as a callee change when its root cause is absent from the trace.

    Returns ``(mutation_to_record, root_in_trace)``. Under the hard profile, a change whose
    function name does not appear in any traceback frame, while the traceback still has at
    least one frame in the project's own source, surfaces in a different function than the
    one that was changed: ``KIND_CALLEE``. The text recipe is unchanged.
    """
    frames = _TRACE_FRAME.findall(stack_trace)
    root_in_trace = any(name == info.name for _, name in frames)
    if not PROFILE.hard or root_in_trace:
        return mutation, root_in_trace
    in_source = any(
        path.startswith(repo.source_prefixes) and not path.startswith(repo.skip_prefixes)
        for path, _ in frames
    )
    if not in_source:
        return mutation, root_in_trace
    return replace(mutation, kind=KIND_CALLEE), root_in_trace


def generate_for_repo(
    ws: Workspace,
    repo: SourceRepo,
    *,
    seed: int,
    target: int,
    per_function: int | None = None,
    per_file: int | None = None,
    max_attempts: int | None = None,
    mutation_set: str = "mixed",
    log: Log = _log,
) -> tuple[list[GeneratedCase], dict[str, Any]]:
    """Generate up to ``target`` valid cases for ``repo`` from its work copy.

    ``mutation_set`` is ``"mixed"`` (all kinds the active profile enumerates, hard kinds
    favoured) or ``"hard"`` (only the v0.3 hard kinds; profile v03 only).
    """
    per_function = per_function or PROFILE.per_function
    work = ws.work(repo)
    started = time.monotonic()
    base = baseline(ws, repo)
    passed = {n for n, o in base.outcomes.items() if o == "passed"}
    report: dict[str, Any] = {
        "repo": repo.slug,
        "role": repo.role,
        "baseline_passed_tests": len(passed),
        "baseline_other_outcomes": len(base.outcomes) - len(passed),
        "attempted": 0,
        "accepted": 0,
        "not_detected_by_tests": 0,
        "rejected_log": 0,
        "rejected_flaky": 0,
        "root_absent_from_trace": 0,
        "by_kind": {},
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
    for file in source_files(repo, work):
        try:
            for info in functions_of(file):
                if (file, info.name) in callers:
                    plan.append((file, info))
        except UnicodeDecodeError:
            continue
    rng.shuffle(plan)
    file_cap = per_file or max(6, math.ceil(target / 3))
    attempts_cap = max_attempts or target * 12
    per_file_count: Counter[str] = Counter()
    cases: list[GeneratedCase] = []
    kinds: Counter[str] = Counter()  # text-shape kinds, used only to balance sampling
    final_kinds: Counter[str] = Counter()  # recorded kinds (may be relabelled as callee changes)
    for file, info in plan:
        if len(cases) >= target or report["attempted"] >= attempts_cap:
            break
        if per_file_count[file] >= file_cap:
            continue
        mutations = enumerate_mutations(sources[file], info, hard=PROFILE.hard)
        if mutation_set == "hard":
            mutations = [m for m in mutations if m.kind in HARD_KINDS]
        rng.shuffle(mutations)
        # Prefer rarer kinds (stable); hard kinds count half so they are drawn about twice as often.
        mutations.sort(key=lambda m: kinds[m.kind] / (2 if m.kind in HARD_KINDS else 1))
        used: set[str] = set()
        for mutation in mutations:
            if len(used) >= per_function or len(cases) >= target:
                break
            if mutation.kind in used or report["attempted"] >= attempts_cap:
                continue
            report["attempted"] += 1
            outcome = evaluate_mutation(
                ws, repo, file, info, mutation, sources[file], callers[(file, info.name)], passed
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
            recorded, root_in_trace = classify_surfacing(repo, info, mutation, log_text[1])
            record = build_record(
                repo,
                len(cases),
                file,
                info,
                recorded,
                node,
                pre,
                post,
                log_text,
                negatives,
                exception,
                base_kind=mutation.kind if recorded.kind != mutation.kind else None,
            )
            entry = _mutation_dict(file, recorded)
            entry.update(base_kind=mutation.kind, root_in_trace=root_in_trace)
            cases.append(GeneratedCase(record, entry))
            used.add(mutation.kind)
            kinds[mutation.kind] += 1
            final_kinds[recorded.kind] += 1
            if not root_in_trace:
                report["root_absent_from_trace"] += 1
            per_file_count[file] += 1
        if cases and len(cases) % 10 == 0:
            log(f"{repo.key}: {len(cases)}/{target} (attempted {report['attempted']})")
    report["accepted"] = len(cases)
    report["by_kind"] = dict(final_kinds)
    report["seconds"] = round(time.monotonic() - started, 1)
    return cases, report


def evaluate_mutation(
    ws: Workspace,
    repo: SourceRepo,
    file: str,
    info: FunctionInfo,
    mutation: Mutation,
    source: str,
    candidate_tests: list[str],
    passed: set[str],
) -> str | tuple[str, PytestRun, PytestRun, tuple[str, str]]:
    """Apply ``mutation`` in the work copy, run the covering tests, always restore the file.

    Returns a report counter name on rejection, or ``(node, post, pre, (log, trace))``.
    """
    work = ws.work(repo)
    path = work / file
    original = path.read_bytes()
    tests = sorted(set(candidate_tests) & passed)[:MAX_TESTS_PER_MUTATION]
    skip = tuple({*repo.skip_prefixes, *(f"{t}/" for t in repo.test_args)})
    try:
        path.write_text(apply_mutation(source, mutation), encoding="utf-8")
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


def _mutation_dict(file: str, mutation: Mutation) -> dict[str, Any]:
    return {
        "file": file,
        "function": mutation.function,
        "kind": mutation.kind,
        "line_no": mutation.line_no,
        "old": mutation.old,
        "new": mutation.new,
        "in_place": mutation.in_place,
    }


def build_record(
    repo: SourceRepo,
    index: int,
    file: str,
    info: FunctionInfo,
    mutation: Mutation,
    node: str,
    pre: PytestRun,
    post: PytestRun,
    log_text: tuple[str, str],
    negatives: list[tuple[str, FunctionInfo, str]],
    exception: str,
    *,
    base_kind: str | None = None,
) -> dict[str, Any]:
    names = [f"{nfile}:{ninfo.name}" for nfile, ninfo, _ in negatives]
    rationale_mutation = (
        replace(mutation, kind=f"{base_kind} (a callee change whose failing frame is elsewhere)")
        if base_kind
        else mutation
    )
    return {
        "record_id": f"{PROFILE.name}-{repo.key}-{index:04d}",
        "dataset_id": PROFILE.train_dataset_id if repo.role == "train" else PROFILE.dev_dataset_id,
        "source_repository": repo.slug,
        "source_commit": repo.commit,
        "source_license": repo.license,
        "language": "python",
        "injection_method": mutation.kind,
        "injection_location": _location(file, info),
        "has_fix_patch": True,
        "test_command": f"python -m pytest {node} --tb=native -q",
        "pre_change_result": {
            "status": "passed",
            "exit_code": pre.returncode,
            "duration_ms": pre.duration_ms,
        },
        "post_change_result": {
            "status": "failed",
            "exit_code": post.returncode,
            "duration_ms": post.duration_ms,
        },
        "error_log": log_text[0],
        "stack_trace": log_text[1],
        "candidates": [
            {"location": _location(file, info), "label": "positive"},
            *(
                {"location": _location(nfile, ninfo), "label": "negative"}
                for nfile, ninfo, _ in negatives
            ),
        ],
        "label_rationale": _rationale(file, rationale_mutation, exception, node, names),
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "generator_version": PROFILE.generator_version,
    }


# -------------------------------------------------------------------------- finalize


def _schema_validator() -> Any:
    from jsonschema import Draft202012Validator, FormatChecker

    schema_path = Path(__file__).resolve().parents[2] / "datasets/schemas/record.schema.json"
    return Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8")), format_checker=FormatChecker()
    )


def validate_cases(cases: list[GeneratedCase]) -> tuple[list[GeneratedCase], dict[str, Any]]:
    """Schema + semantic gates + duplicate gates; returns survivors and an aggregate report."""
    validator = _schema_validator()
    records = [c.record for c in cases]
    schema_errors = [len(list(validator.iter_errors(r))) for r in records]
    results = validate_record_set(records, redistributed=True)
    reasons: Counter[str] = Counter()
    seen_edits: set[tuple[str, str, str, str]] = set()
    kept: list[GeneratedCase] = []
    duplicate_edits = 0
    for case, errors, result in zip(cases, schema_errors, results, strict=True):
        if errors:
            reasons["schema-violation"] += 1
            continue
        if not result.passed:
            for reason in result.rejection_reasons:
                reasons[reason.value] += 1
            continue
        edit = (
            case.record["source_repository"],
            case.mutation["file"],
            case.mutation["function"],
            case.mutation["new"],
        )
        if edit in seen_edits:
            duplicate_edits += 1
            continue
        seen_edits.add(edit)
        kept.append(case)
    return kept, {
        "input": len(cases),
        "kept": len(kept),
        "schema_violations": sum(schema_errors),
        "semantic_rejections": dict(reasons),
        "duplicate_edits": duplicate_edits,
    }


def _renumber(cases: list[GeneratedCase]) -> list[GeneratedCase]:
    """Stable, collision-free record ids per repository after filtering."""
    counters: Counter[str] = Counter()
    for case in cases:
        record_id: str = case.record["record_id"]
        prefix = f"{PROFILE.name}-"
        key = record_id[len(prefix) : record_id.rfind("-")]
        case.record["record_id"] = f"{prefix}{key}-{counters[key]:04d}"
        counters[key] += 1
    return cases


def write_dev_trees(
    ws: Workspace,
    cases: list[GeneratedCase],
    out_dir: Path,
    resolve: Callable[[str], SourceRepo] = get_repo,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one failing-state tree per dev case plus the evaluator manifests."""
    from failurelens.config import IndexConfig
    from failurelens.indexing.repository import iter_source_files

    trees_root = ws.root / "trees"
    snapshots: dict[str, Any] = {}
    cases_manifest: dict[str, Any] = {}
    repositories: dict[str, Any] = {}
    for case in cases:
        record, mutation = case.record, case.mutation
        repo = resolve(record["source_repository"])
        clone = ws.source(repo)
        root = trees_root / record["record_id"]
        copy_tracked_files(clone, root)
        target = root / mutation["file"]
        clean = target.read_text(encoding="utf-8")
        target.write_text(
            apply_mutation(
                clean, Mutation(**{k: mutation[k] for k in Mutation.__dataclass_fields__})
            ),
            encoding="utf-8",
        )
        indexed = {
            relative: sha256_file(path) for path, relative in iter_source_files(root, IndexConfig())
        }
        cases_manifest[record["record_id"]] = {
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
    write_private_json(out_dir / "tree_manifest.json", {"cases": cases_manifest})
    return repositories, cases_manifest


def finalize(ws: Workspace, log: Log = _log) -> dict[str, Any]:
    """Combine raw per-repository output into the train and dev corpora."""
    validate_table()
    by_role: dict[str, list[GeneratedCase]] = {"train": [], "dev": []}
    repo_reports: list[dict[str, Any]] = []
    for repo in active_repositories():
        raw_path = ws.raw(repo)
        if not raw_path.exists():
            log(f"skip {repo.key}: no raw output")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        repo_reports.append(raw["report"])
        by_role[repo.role].extend(GeneratedCase(c["record"], c["mutation"]) for c in raw["cases"])
    summary: dict[str, Any] = {
        "corpus_id": PROFILE.corpus_id,
        "generator_version": PROFILE.generator_version,
    }
    digests: dict[str, str] = {}
    for role, dataset_id in (
        ("train", PROFILE.train_dataset_id),
        ("dev", PROFILE.dev_dataset_id),
    ):
        kept, validation = validate_cases(by_role[role])
        kept = _renumber(kept)
        out_dir = ws.root / role
        records = [c.record for c in kept]
        write_private_json(out_dir / "records.json", records)
        write_private_json(
            out_dir / "mutations.json", {c.record["record_id"]: c.mutation for c in kept}
        )
        digests[role] = sha256_file(out_dir / "records.json")
        repos = sorted({c.record["source_repository"] for c in kept})
        summary[role] = {
            "dataset_id": dataset_id,
            "records": len(records),
            "repositories": repos,
            "records_by_repository": dict(Counter(c.record["source_repository"] for c in kept)),
            "by_mutation": dict(Counter(c.record["injection_method"] for c in kept)),
            "hard_records": (hard := sum(c.record["injection_method"] in HARD_KINDS for c in kept)),
            "hard_share": round(hard / len(kept), 4) if kept else 0.0,
            "root_absent_from_trace": sum(not c.mutation.get("root_in_trace", True) for c in kept),
            "validation": validation,
            "records_sha256": digests[role],
        }
        if role == "dev" and kept:
            write_dev_trees(ws, kept, out_dir)
    train_repos = summary["train"]["repositories"]
    dev_repos = summary["dev"]["repositories"]
    overlap = (set(train_repos) | set(LEGACY_TRAIN_REPOSITORIES)) & set(dev_repos)
    held = HOLDOUT_REPOSITORIES & (set(train_repos) | set(dev_repos))
    if overlap or held:
        raise RuntimeError("repository overlap between train, dev or hold-out")
    write_private_json(
        ws.root / "dev" / "train_manifest.json",
        {
            "approval_id": PROFILE.approval_id,
            "training_repositories": sorted(set(train_repos) | set(LEGACY_TRAIN_REPOSITORIES)),
        },
    )
    corpus_digest = canonical_digest({"train": digests["train"], "dev": digests["dev"]})
    summary.update(
        corpus_sha256=corpus_digest,
        train_dev_repository_overlap=0,
        holdout_repository_overlap=0,
        repositories=repo_reports,
    )
    write_private_json(
        ws.root / "manifest.json",
        {
            "corpus_id": PROFILE.corpus_id,
            "generator_version": PROFILE.generator_version,
            "purpose": "private offline synthetic training and model-selection only",
            "records_sha256": digests,
            "corpus_sha256": corpus_digest,
            "pins": {
                r.slug: r.commit
                for r in active_repositories()
                if r.slug in set(train_repos + dev_repos)
            },
            "excluded_holdout_repositories": sorted(HOLDOUT_REPOSITORIES),
        },
    )
    write_private_json(ws.root / "report.json", summary)
    return summary


# ------------------------------------------------------------------------------ CLI


def _generate_one(args: argparse.Namespace, ws: Workspace, repo: SourceRepo) -> None:
    raw_path = ws.raw(repo)
    if raw_path.exists() and not args.force:
        _log(f"skip {repo.key}: raw output exists")
        return
    target = args.train_target if repo.role == "train" else args.dev_target
    cases, report = generate_for_repo(
        ws,
        repo,
        seed=args.seed if args.seed is not None else PROFILE.seed,
        target=target,
        mutation_set=args.mutation_set,
    )
    write_private_json(
        raw_path,
        {"report": report, "cases": [{"record": c.record, "mutation": c.mutation} for c in cases]},
    )
    _log(f"done {repo.key}: {json.dumps({k: v for k, v in report.items() if k != 'by_kind'})}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["prepare", "generate", "finalize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="virtualenv python with pytest")
    parser.add_argument("--repos", nargs="*", default=None, help="subset of repository keys")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="v02")
    parser.add_argument("--mutation-set", choices=["mixed", "hard"], default="mixed")
    parser.add_argument("--seed", type=int, default=None, help="default: the profile's seed")
    parser.add_argument("--train-target", type=int, default=32, help="cases per train repository")
    parser.add_argument("--dev-target", type=int, default=16, help="cases per dev repository")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    ws = Workspace(args.root.expanduser(), args.python.expanduser())
    ws.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    ws.root.chmod(0o700)
    set_profile(args.profile)
    repos = [r for r in active_repositories() if args.repos is None or r.key in args.repos]
    if args.stage == "prepare":
        evidence = {r.slug: prepare_repo(ws, r) for r in repos}
        write_private_json(ws.root / "license_evidence.json", evidence)
        print(json.dumps(evidence, indent=2))
    elif args.stage == "generate":
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda r: _generate_one(args, ws, r), repos))
    else:
        print(json.dumps(finalize(ws), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GeneratedCase",
    "Workspace",
    "build_record",
    "canonical_digest",
    "choose_negatives",
    "classify_license",
    "evaluate_mutation",
    "finalize",
    "generate_for_repo",
    "main",
    "prepare_repo",
    "sha256_file",
    "validate_cases",
]
