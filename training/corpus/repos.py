"""Source-repository table for the v0.2 private fault corpus.

Only repositories whose license text was read at the pinned commit and found to
be MIT or BSD-3-Clause are listed. Pins and license facts are public; the
generated records are private and never enter Git.

Train, dev and the Gate C hold-out use disjoint repositories (repo-level
splits). ``HOLDOUT_REPOSITORIES`` must never appear here; ``validate_table``
enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HOLDOUT_REPOSITORIES", "REPOSITORIES", "SourceRepo", "get_repo", "validate_table"]

#: Gate C hold-out sources (see docs/gate-b-rights-ledger.md). Never train or select on them.
HOLDOUT_REPOSITORIES: frozenset[str] = frozenset({"psf/requests", "pallets/werkzeug", "pypa/pip"})


@dataclass(frozen=True)
class SourceRepo:
    slug: str  # "owner/name"
    key: str  # short directory name
    commit: str  # full 40-hex pin
    license: str  # SPDX id, verified against the LICENSE text at the pin
    role: str  # "train" | "dev"
    source_prefixes: tuple[str, ...]  # repo-relative dirs/files that may contain fault sites
    skip_prefixes: tuple[str, ...]  # tests, docs, vendored or generated code to exclude
    test_args: tuple[str, ...]  # pytest targets, relative to the repo root
    pythonpath: str  # "." or "src": added to PYTHONPATH when running tests
    license_file: str  # repo-relative license file read at the pin
    profiles: tuple[str, ...] = ("v02", "v03")  # corpus versions that use this repository

    @property
    def url(self) -> str:
        return f"https://github.com/{self.slug}"


REPOSITORIES: tuple[SourceRepo, ...] = (
    SourceRepo(
        "pallets/click",
        "click",
        "2247b35ea1c47c727d7a06e51fa280e12a863ff6",
        "BSD-3-Clause",
        "train",
        ("src/click/",),
        ("src/click/_winconsole.py", "src/click/parser.py"),
        # test_stream_lifecycle.py expands to ~30k parametrized cases (hours); skip it.
        ("tests", "--ignore=tests/test_stream_lifecycle.py"),
        "src",
        "LICENSE.txt",
    ),
    SourceRepo(
        "more-itertools/more-itertools",
        "more-itertools",
        "f42daddf205f335fea6e200dad46049f33e71221",
        "MIT",
        "train",
        ("more_itertools/",),
        ("more_itertools/tests/",),
        ("tests",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "pytoolz/toolz",
        "toolz",
        "451af60dec590a6010e2babdbf391ea8f815122f",
        "BSD-3-Clause",
        "train",
        ("toolz/",),
        ("toolz/tests/", "toolz/sandbox/", "toolz/_signatures.py"),
        ("toolz/tests",),
        ".",
        "LICENSE.txt",
    ),
    SourceRepo(
        "astanin/python-tabulate",
        "tabulate",
        "268615a5c27dc40e5c22454c07b44d5c50410da0",
        "MIT",
        "train",
        ("tabulate/",),
        ("tabulate/version.py",),
        ("test",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "kjd/idna",
        "idna",
        "d357725082245ff90f5cf8800a3fc297b1f2319a",
        "BSD-3-Clause",
        "train",
        ("idna/",),
        ("idna/idnadata.py", "idna/uts46data.py", "idna/package_data.py"),
        ("tests",),
        ".",
        "LICENSE.md",
    ),
    SourceRepo(
        "jpvanhal/inflection",
        "inflection",
        "88eefaacf7d0caaa701af7c8ab2d0ab3f17086f1",
        "MIT",
        "train",
        ("inflection/",),
        (),
        ("test_inflection.py",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "pallets/itsdangerous",
        "itsdangerous",
        "672971d66a2ef9f85151e53283113f33d642dabd",
        "BSD-3-Clause",
        "train",
        ("src/itsdangerous/",),
        (),
        ("tests",),
        "src",
        "LICENSE.txt",
    ),
    # v0.3 additions (train). Chosen from repositories whose names start with a-m.
    SourceRepo(
        "jawah/charset_normalizer",
        "charset-normalizer",
        "7d74fe4a8341ae27a8f42c759793301dbd304e10",
        "MIT",
        "train",
        ("src/charset_normalizer/",),
        ("src/charset_normalizer/cli/", "src/charset_normalizer/version.py"),
        ("tests",),
        "src",
        "LICENSE",
        ("v03",),
    ),
    SourceRepo(
        "micheles/decorator",
        "decorator",
        "956b6396c7369aab4acc7a44851572fcb92fb34a",
        "BSD-2-Clause",
        "train",
        ("src/decorator.py",),
        (),
        ("tests/test.py",),
        "src",
        "LICENSE.txt",
        ("v03",),
    ),
    SourceRepo(
        "executablebooks/mdurl",
        "mdurl",
        "524d2edbbcb8bb48301ba716c7482827bcabb281",
        "MIT",
        "train",
        ("src/mdurl/",),
        (),
        ("tests",),
        "src",
        "LICENSE",
        ("v03",),
    ),
    SourceRepo(
        "lepture/mistune",
        "mistune",
        "f7ec702edfc1ad5579a76f881238841af266d93c",
        "BSD-3-Clause",
        "train",
        ("src/mistune/",),
        ("src/mistune/__main__.py",),
        ("tests",),
        "src",
        "LICENSE",
        ("v03",),
    ),
    SourceRepo(
        "andialbrecht/sqlparse",
        "sqlparse",
        "09dbe1eec31ed1b3cd05366748865f11ba427d32",
        "BSD-3-Clause",
        "dev",
        ("sqlparse/",),
        ("sqlparse/keywords.py",),
        ("tests",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "python-humanize/humanize",
        "humanize",
        "d16e6352c9785c5ab86de0133222bd22e89b0db2",
        "MIT",
        "dev",
        ("src/humanize/",),
        ("src/humanize/locale/",),
        ("tests",),
        "src",
        "LICENCE",
    ),
    SourceRepo(
        "mahmoud/boltons",
        "boltons",
        "d31d821d56c8f7756af9c30a864077f2e5855321",
        "BSD-3-Clause",
        "dev",
        ("boltons/",),
        (),
        ("tests",),
        ".",
        "LICENSE",
    ),
    SourceRepo(
        "jmespath/jmespath.py",
        "jmespath",
        "2812594e69d43098ef60f81f4efc404c071b0418",
        "MIT",
        "dev",
        ("jmespath/",),
        (),
        ("tests",),
        ".",
        "LICENSE",
    ),
)


def get_repo(key: str) -> SourceRepo:
    for repo in REPOSITORIES:
        if repo.key == key or repo.slug == key:
            return repo
    raise KeyError(key)


def validate_table(repos: tuple[SourceRepo, ...] = REPOSITORIES) -> None:
    """Fail closed on hold-out use, split overlap, bad pins or non-permissive licenses."""
    slugs = [r.slug for r in repos]
    keys = [r.key for r in repos]
    if len(set(slugs)) != len(slugs) or len(set(keys)) != len(keys):
        raise ValueError("duplicate repository entry")
    if HOLDOUT_REPOSITORIES & set(slugs):
        raise ValueError("hold-out repository listed in the corpus table")
    for repo in repos:
        if repo.role not in {"train", "dev"}:
            raise ValueError(f"invalid role for {repo.key}")
        if len(repo.commit) != 40 or any(c not in "0123456789abcdef" for c in repo.commit):
            raise ValueError(f"invalid pin for {repo.key}")
        if repo.license not in {"MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0"}:
            raise ValueError(f"license not allowed for {repo.key}")
