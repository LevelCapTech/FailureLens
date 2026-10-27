"""Configuration objects for indexing and diagnosis.

Defaults are chosen so the baseline runs with **no required configuration** and
**no network access**. Everything here is plain data (Pydantic models) to keep
behaviour explicit, serializable, and easy to test.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

#: Directory names pruned during repository traversal (virtualenvs, caches, VCS, ...).
DEFAULT_EXCLUDE_DIRS: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "build",
    "dist",
    ".eggs",
    ".idea",
    ".vscode",
    "site-packages",
)

#: File suffixes indexed by default. Python is supported first; other languages
#: are intentionally out of scope for the baseline indexer (see docs/roadmap.md).
DEFAULT_INCLUDE_SUFFIXES: tuple[str, ...] = (".py",)


class IndexConfig(BaseModel):
    """Controls which files are enumerated and parsed during indexing."""

    model_config = ConfigDict(extra="forbid")

    include_suffixes: tuple[str, ...] = Field(default=DEFAULT_INCLUDE_SUFFIXES)
    exclude_dirs: tuple[str, ...] = Field(default=DEFAULT_EXCLUDE_DIRS)
    exclude_globs: tuple[str, ...] = Field(
        default=(),
        description="fnmatch patterns (matched against repo-relative POSIX paths) to exclude.",
    )
    max_file_bytes: int = Field(
        default=1_000_000, ge=0, description="Skip files larger than this (0 disables the limit)."
    )
    follow_symlinks: bool = Field(default=False)


class DiagnoseConfig(BaseModel):
    """Controls a single diagnosis run."""

    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1, description="Maximum number of candidates to return.")
    index: IndexConfig = Field(default_factory=IndexConfig)
    read_candidate_source: bool = Field(
        default=True,
        description="Read candidate source spans to compute lexical-match evidence.",
    )
