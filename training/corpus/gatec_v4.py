"""Private "Gate C v4" evaluation-set builder.

Gate C v3 is consumed (the v0.5 independent confirmation looked at its aggregate
numbers). Gate C v4 is a new private, repository-disjoint final-evaluation set built
with exactly the same generator, categories and selection as Gate C v3
(:mod:`training.corpus.gatec_v3`), on a fresh repository table that is checked to be
disjoint from every repository used before, including the v3 sources and the
repositories that were tried and dropped for v3. Third-party test code executes only
inside the network-less container (``--python training/corpus/sandbox/sandbox_python.sh``).

No model or baseline output is consulted anywhere; selection uses structural
metadata only. Stages are the same as for v3::

    python -m training.corpus.gatec_v4 --root DIR --python SANDBOX prepare
    python -m training.corpus.gatec_v4 --root DIR --python SANDBOX generate
    python -m training.corpus.gatec_v4 --root DIR --python SANDBOX finalize
    python -m training.corpus.gatec_v4 --root DIR --python SANDBOX manifest --train-repos-file F
    python -m training.corpus.gatec_v4 --root DIR --python SANDBOX check
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.corpus import build, gatec_v2, gatec_v3
from training.corpus.build import Workspace, write_private_json
from training.corpus.gatec_v3 import CATEGORIES, V3_REPOSITORIES
from training.corpus.repos import SourceRepo

__all__ = ["V4_REPOSITORIES", "validate_v4_table"]

V4_DATASET_ID = "failurelens-gatec-v4"
V4_ID = "failurelens-gatec-v4"
V4_GENERATOR_VERSION = "gatec-v4-1.0"
DEFAULT_SEED = 20261110

#: Repositories tried and dropped while building v3 (infrastructure reasons only).
V3_DROPPED_REPOSITORIES: tuple[str, ...] = (
    "python-semver/python-semver",
    "scrapy/w3lib",
    "python-hyper/hyperframe",
    "tobgu/pyrsistent",
    "tartley/colorama",
)
V3_SOURCE_SLUGS: tuple[str, ...] = tuple(r.slug for r in V3_REPOSITORIES)
EARLIER_V3_SLUGS: tuple[str, ...] = (*V3_SOURCE_SLUGS, *V3_DROPPED_REPOSITORIES)

V4_REPOSITORIES: tuple[SourceRepo, ...] = (
    SourceRepo(
        "pyparsing/pyparsing",
        "pyparsing",
        "810199f89bf916590426e1c5cb3d612584f11a2f",
        "MIT",
        "eval",
        ("pyparsing/",),
        (),
        (
            "tests",
            "--ignore=tests/test_diagram.py",
            "--ignore=tests/test_matplotlib_cases.py",
        ),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "hukkin/tomli",
        "tomli",
        "43a86ade8b795c6ef9b4ec53372a607cd1ed426f",
        "MIT",
        "eval",
        ("src/tomli/",),
        (),
        ("tests",),
        "src",
        "LICENSE",
    ),
    SourceRepo(
        "r1chardj0n3s/parse",
        "parse",
        "616ac26ca4077266dee2479dc51a8d19feab0ada",
        "MIT",
        "eval",
        ("parse/",),
        (),
        ("tests",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "keleshev/schema",
        "schema",
        "310a1239b62f500284ce3bd91b7dabf70467f23e",
        "MIT",
        "eval",
        ("schema/",),
        (),
        ("test_schema.py",),
        ".",
        "LICENSE-MIT",
    ),
    SourceRepo(
        "martinblech/xmltodict",
        "xmltodict",
        "6e29fba282e68631034dd2722a413b4e35276584",
        "MIT",
        "eval",
        ("xmltodict.py",),
        (),
        ("tests",),
        ".",
        "LICENSE",
    ),
)

#: Tried and dropped while building v4, on infrastructure grounds and before any model or
#: baseline output existed: wcwidth (suite needs a C extension), funcy (suite needs an
#: absent helper package), sortedcontainers (no failure log with a usable traceback).
V4_DROPPED_REPOSITORIES: tuple[str, ...] = (
    "jquast/wcwidth",
    "Suor/funcy",
    "grantjenks/python-sortedcontainers",
)


def validate_v4_table(
    repos: tuple[SourceRepo, ...] = V4_REPOSITORIES,
    extra_train: tuple[str, ...] = (),
) -> None:
    """Fail closed on overlap with any earlier source (including v3) or bad pins."""
    gatec_v3.validate_v3_table(repos, (*extra_train, *EARLIER_V3_SLUGS))


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
    parser.add_argument("--train-repos-file", type=Path, default=None)
    args = parser.parse_args(argv)
    ws = Workspace(args.root.expanduser(), args.python.expanduser())
    ws.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    ws.root.chmod(0o700)
    train_repos = gatec_v3._read_train_repos(args.train_repos_file)
    validate_v4_table(extra_train=tuple(train_repos))
    repos = [r for r in V4_REPOSITORIES if args.repos is None or r.key in args.repos]
    if args.stage == "prepare":
        evidence = {r.slug: build.prepare_repo(ws, r) for r in repos}
        write_private_json(ws.root / "license_evidence.json", evidence)
        print(json.dumps(evidence, indent=2))
    elif args.stage == "generate":
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(
                pool.map(
                    lambda r: gatec_v3._generate_one(
                        args, ws, r, V4_DATASET_ID, V4_GENERATOR_VERSION
                    ),
                    repos,
                )
            )
    elif args.stage == "finalize":
        summary = gatec_v3.finalize(
            ws,
            total=args.total,
            seed=args.seed,
            train_repos=sorted({*train_repos, *EARLIER_V3_SLUGS}),
            repos=V4_REPOSITORIES,
            eval_id=V4_ID,
            generator_version=V4_GENERATOR_VERSION,
            id_prefix="gcv4",
            validate=validate_v4_table,
            extra_excluded=EARLIER_V3_SLUGS,
        )
        print(json.dumps(summary, indent=2))
    elif args.stage == "manifest":
        meta = json.loads((ws.root / "manifest.json").read_text(encoding="utf-8"))
        path = gatec_v2.write_train_manifest(
            ws.root, sorted({*train_repos, *EARLIER_V3_SLUGS}), sorted(meta["pins"])
        )
        print(json.dumps({"train_manifest": path.name}))
    else:
        print(json.dumps(gatec_v2.check(ws, categories=CATEGORIES), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
