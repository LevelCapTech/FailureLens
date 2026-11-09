"""Private "Gate C v3" evaluation-set builder.

Gate C v2 is consumed (its results were looked at). Gate C v3 is a new private,
repository-disjoint final-evaluation set for the independent confirmation of the
call-graph baseline and later candidates. It reuses the sandboxed generation
machinery of :mod:`training.corpus.gatec_v2` and executes third-party test code
only inside the network-less container (``--python training/corpus/sandbox/sandbox_python.sh``).

What is new compared with Gate C v2:

* A fresh repository table, checked to be disjoint from every repository ever used:
  v0.2/v0.3 train and dev, the legacy training repositories, the first hold-out and
  the Gate C v2 sources (plus any caller-supplied extra training repositories).
* The v0.3 hard text shapes (multi-line edits, argument swaps and drops, guard
  edits, callee changes) are enumerated alongside the v2 statement-level edits.
* Four structural surfacing categories, computed from the stack trace and an
  independent name-based static call check (written here, not the library's call
  graph) so the set is not designed around the baseline under test:

  ============================  ==============================================
  ``root-is-innermost-frame``   the changed function raised
  ``root-deeper-in-stack``      the changed function is in the stack, failure
                                surfaces below it or in a test after it
  ``root-absent-call-related``  not in the stack, but directly calls, or is
                                directly called by, a production stack frame
  ``root-absent-unrelated``     not in the stack and no direct call relation
  ============================  ==============================================

* No model output and no baseline output is consulted anywhere. Selection is
  stratified on repository, category and mutation kind with a fixed seed.

Stages (each resumable)::

    python -m training.corpus.gatec_v3 --root DIR --python SANDBOX prepare
    python -m training.corpus.gatec_v3 --root DIR --python SANDBOX generate
    python -m training.corpus.gatec_v3 --root DIR --python SANDBOX finalize
    python -m training.corpus.gatec_v3 --root DIR --python SANDBOX manifest --train-repos-file F
    python -m training.corpus.gatec_v3 --root DIR --python SANDBOX check
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

from training.corpus import build, gatec_v2
from training.corpus.build import GeneratedCase, Workspace, sha256_file, write_private_json
from training.corpus.gatec_v2 import (
    CATEGORY_ABSENT,
    EvalCase,
    classify_surfacing,
    write_eval_trees,
    write_train_manifest,
)
from training.corpus.repos import HOLDOUT_REPOSITORIES, REPOSITORIES, SourceRepo

__all__ = [
    "CATEGORIES",
    "CATEGORY_ABSENT_RELATED",
    "CATEGORY_ABSENT_UNRELATED",
    "V3_REPOSITORIES",
    "classify_v3",
    "default_train_repositories",
    "select_v3",
    "validate_v3_table",
]

V3_DATASET_ID = "failurelens-gatec-v3"
V3_ID = "failurelens-gatec-v3"
V3_GENERATOR_VERSION = "gatec-v3-1.0"
V3_APPROVAL_ID = "owner-approved-gatec-v3-2026-11-09"
DEFAULT_SEED = 20261109
Log = Callable[[str], None]

CATEGORY_INNERMOST = gatec_v2.CATEGORY_INNERMOST
CATEGORY_MID = gatec_v2.CATEGORY_MID
CATEGORY_ABSENT_RELATED = "root-absent-call-related"
CATEGORY_ABSENT_UNRELATED = "root-absent-unrelated"
CATEGORIES = (
    CATEGORY_INNERMOST,
    CATEGORY_MID,
    CATEGORY_ABSENT_RELATED,
    CATEGORY_ABSENT_UNRELATED,
)

#: Target share per category (structural; the achieved mix is reported honestly).
#: Call-related absent cases get the largest share because they are what a call-graph
#: signal can and cannot do; unrelated-absent cases are kept as an honest negative control.
CATEGORY_SHARE: dict[str, float] = {
    CATEGORY_INNERMOST: 0.20,
    CATEGORY_MID: 0.20,
    CATEGORY_ABSENT_RELATED: 0.35,
    CATEGORY_ABSENT_UNRELATED: 0.25,
}

#: Repositories used by the Gate C v2 final evaluation (consumed; never reuse).
GATEC_V2_REPOSITORIES: tuple[str, ...] = tuple(r.slug for r in gatec_v2.EVAL_REPOSITORIES)

V3_REPOSITORIES: tuple[SourceRepo, ...] = (
    SourceRepo(
        "PyCQA/pyflakes",
        "pyflakes",
        "5324946c5d8602697f5df6d98ad162920b3e5d61",
        "MIT",
        "eval",
        ("pyflakes/",),
        ("pyflakes/test/", "pyflakes/scripts/", "pyflakes/__main__.py"),
        ("pyflakes/test",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "pallets/markupsafe",
        "markupsafe",
        "4e111271b1494995708a9e878a10829b5e10c7ba",
        "BSD-3-Clause",
        "eval",
        ("src/markupsafe/",),
        (),
        ("tests",),
        "src",
        "LICENSE.txt",
    ),
    SourceRepo(
        "python-hyper/h11",
        "h11",
        "62c5068c971579d61fa1b55373390e12f25fd856",
        "MIT",
        "eval",
        ("h11/",),
        ("h11/tests/",),
        ("h11/tests",),
        ".",
        "LICENSE.txt",
    ),
    SourceRepo(
        "python-hyper/hpack",
        "hpack",
        "6aa8c0b2265f558f7184047aa09dcc573e625f82",
        "MIT",
        "eval",
        ("src/hpack/",),
        (),
        ("tests",),
        "src",
        "LICENSE",
    ),
    SourceRepo(
        "Python-Markdown/markdown",
        "markdown",
        "40eca048f57456d855a491e7f94ba215594f3f57",
        "BSD-3-Clause",
        "eval",
        ("markdown/",),
        (),
        (
            "tests",
            # test_apis needs PyYAML (absent from the sandbox image); the md_in_html module
            # turns a pytest collection warning into an error. Pre-existing failing tests
            # in the baseline are simply not used as detectors.
            "--ignore=tests/test_apis.py",
            "--ignore=tests/test_syntax/extensions/test_md_in_html.py",
            "-ofilterwarnings=ignore",
        ),
        ".",
        "LICENSE.md",
    ),
)


# ----------------------------------------------------------------------- table checks


def validate_v3_table(
    repos: tuple[SourceRepo, ...] = V3_REPOSITORIES,
    extra_train: tuple[str, ...] = (),
) -> None:
    """Fail closed on any overlap with every repository ever used, or bad pins."""
    slugs = [r.slug for r in repos]
    keys = [r.key for r in repos]
    if len(set(slugs)) != len(slugs) or len(set(keys)) != len(keys):
        raise ValueError("duplicate repository entry")
    used = {r.slug.lower() for r in REPOSITORIES}
    used |= {s.lower() for s in gatec_v2.LEGACY_TRAIN_REPOSITORIES}
    used |= {s.lower() for s in HOLDOUT_REPOSITORIES}
    used |= {s.lower() for s in GATEC_V2_REPOSITORIES}
    used |= {s.lower() for s in extra_train}
    used_keys = {r.key for r in REPOSITORIES} | {r.key for r in gatec_v2.EVAL_REPOSITORIES}
    for repo in repos:
        if repo.slug.lower() in used or repo.key in used_keys:
            raise ValueError(f"repository overlaps an earlier source: {repo.key}")
        if repo.role != "eval":
            raise ValueError(f"invalid role for {repo.key}")
        if len(repo.commit) != 40 or any(c not in "0123456789abcdef" for c in repo.commit):
            raise ValueError(f"invalid pin for {repo.key}")
        if repo.license not in {"MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0"}:
            raise ValueError(f"license not allowed for {repo.key}")


def default_train_repositories() -> list[str]:
    """Every repository used to train or tune any earlier candidate (train and dev)."""
    return sorted({r.slug for r in REPOSITORIES} | set(gatec_v2.LEGACY_TRAIN_REPOSITORIES))


# ------------------------------------------------------------------ surfacing category

_FRAME = re.compile(r'^\s*File "([^"]+)", line \d+, in (\S+)', re.MULTILINE)
_TEST_PATH = gatec_v2._TEST_PATH


def _callee_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            target = child.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


@lru_cache(maxsize=64)
def _static_graph(
    work: str, files: tuple[str, ...]
) -> tuple[dict[tuple[str, str], set[str]], Counter[str]]:
    """Name-based direct-call edges and definition counts of the clean source tree.

    Independent of the library's own call graph on purpose: it only answers "does
    function A's body call something named like B", so categories do not depend on
    the code under evaluation.
    """
    edges: dict[tuple[str, str], set[str]] = {}
    defs: Counter[str] = Counter()
    for rel in files:
        path = Path(work) / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name] += 1
                edges.setdefault((rel, node.name), set()).update(_callee_names(node))
    return edges, defs


def classify_v3(
    stack_trace: str,
    file: str,
    function: str,
    edges: dict[tuple[str, str], set[str]],
    defs: Counter[str],
) -> str:
    """Structural category of a failing case (stack trace text + a name-based call check).

    ``root-absent-call-related``: the changed function is absent from the stack but,
    within one hop, a production stack frame calls it or it calls a production stack
    frame, where the relation is by a name that is defined exactly once in the
    repository (so a common name like ``get`` cannot create a spurious relation).
    """
    base = classify_surfacing(stack_trace, file, function)
    if base != CATEGORY_ABSENT:
        return base
    frames = [(p, n) for p, n in _FRAME.findall(stack_trace) if not _TEST_PATH.search(p)]
    frames = [f for f in frames if f in edges]
    unique_root = defs[function] == 1
    root_calls = edges.get((file, function), set())
    for frame in frames:
        if unique_root and function in edges.get(frame, set()):
            return CATEGORY_ABSENT_RELATED
        if defs[frame[1]] == 1 and frame[1] in root_calls:
            return CATEGORY_ABSENT_RELATED
    return CATEGORY_ABSENT_UNRELATED


def _make_categorizer(repo: SourceRepo) -> Callable[[str, str, str, Path], str]:
    def categorize(stack_trace: str, file: str, function: str, work: Path) -> str:
        files = tuple(build.source_files(repo, work))
        edges, defs = _static_graph(str(work), files)
        return classify_v3(stack_trace, file, function, edges, defs)

    return categorize


# ------------------------------------------------------------------------------ selection


def select_v3(
    cases: list[EvalCase], total: int, *, seed: int, per_function: int = 2
) -> list[EvalCase]:
    """Seeded, stratified selection (repository, category, kind); structural metadata only."""
    return gatec_v2.select_cases(
        cases, total, seed=seed, per_function=per_function, shares=CATEGORY_SHARE
    )


# --------------------------------------------------------------------------- finalize


def _renumber(cases: list[EvalCase]) -> None:
    counters: Counter[str] = Counter()
    for case in cases:
        key = case.repo.split("/", 1)[1].lower().replace(".", "-")
        case.record["record_id"] = f"gcv3-{key}-{counters[key]:04d}"
        counters[key] += 1


def finalize(
    ws: Workspace,
    *,
    total: int,
    seed: int,
    train_repos: list[str],
    log: Log = build._log,
) -> dict[str, Any]:
    validate_v3_table(extra_train=tuple(train_repos))
    pool: list[EvalCase] = []
    repo_reports: list[dict[str, Any]] = []
    for repo in V3_REPOSITORIES:
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
    selected = select_v3(survivors, total, seed=seed)
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
                "multi_line": "\n" in c.mutation["old"],
            }
            for c in selected
        },
    )
    by_slug = {r.slug: r for r in V3_REPOSITORIES}
    write_eval_trees(ws, selected, out_dir, lambda slug: by_slug[slug])
    eval_repos = sorted({c.repo for c in selected})
    write_train_manifest(out_dir, train_repos, eval_repos)
    digest = sha256_file(out_dir / "records.json")
    summary: dict[str, Any] = {
        "eval_id": V3_ID,
        "generator_version": V3_GENERATOR_VERSION,
        "records": len(records),
        "repositories": eval_repos,
        "records_by_repository": dict(Counter(c.repo for c in selected)),
        "by_mutation": dict(Counter(c.kind for c in selected)),
        "by_surfacing_category": dict(Counter(c.category for c in selected)),
        "candidate_pool_size": len(pool),
        "candidate_pool_by_category": dict(Counter(c.category for c in pool)),
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
            "eval_id": V3_ID,
            "generator_version": V3_GENERATOR_VERSION,
            "purpose": "private final independent evaluation; one use per selected configuration",
            "records_sha256": digest,
            "pins": {r.slug: r.commit for r in V3_REPOSITORIES if r.slug in set(eval_repos)},
            "excluded_repositories": sorted(
                {
                    *HOLDOUT_REPOSITORIES,
                    *(r.slug for r in REPOSITORIES),
                    *gatec_v2.LEGACY_TRAIN_REPOSITORIES,
                    *GATEC_V2_REPOSITORIES,
                }
            ),
        },
    )
    write_private_json(out_dir / "report.json", summary)
    return summary


# ------------------------------------------------------------------------------ CLI


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
    cases, report = gatec_v2.generate_eval_repo(
        ws,
        repo,
        seed=args.seed,
        target=args.repo_target,
        per_function=3,
        hard=True,
        dataset_id=V3_DATASET_ID,
        generator_version=V3_GENERATOR_VERSION,
        categorize=_make_categorizer(repo),
    )
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
    parser.add_argument("--repo-target", type=int, default=30, help="oversupply per repository")
    parser.add_argument("--total", type=int, default=60, help="final number of cases")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--train-repos-file",
        type=Path,
        default=None,
        help="JSON list (or {training_repositories: [...]}) of extra training repositories "
        "to add to the v0.2/v0.3 train, dev and legacy set in train_manifest.json",
    )
    args = parser.parse_args(argv)
    ws = Workspace(args.root.expanduser(), args.python.expanduser())
    ws.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    ws.root.chmod(0o700)
    train_repos = _read_train_repos(args.train_repos_file)
    validate_v3_table(extra_train=tuple(train_repos))
    repos = [r for r in V3_REPOSITORIES if args.repos is None or r.key in args.repos]
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
        path = write_train_manifest(ws.root, train_repos, sorted(meta["pins"]))
        print(json.dumps({"train_manifest": path.name, "training_repositories": len(train_repos)}))
    else:
        print(json.dumps(gatec_v2.check(ws, categories=CATEGORIES), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
