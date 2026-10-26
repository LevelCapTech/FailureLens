"""Repository traversal and index construction.

Enumerates source files under a repository root (pruning common non-source
directories and honouring user excludes), then dispatches each file to a
language-specific indexer. Only Python is indexed by the baseline; the traversal
and dispatch are structured so additional languages can be added later.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from failurelens.config import IndexConfig
from failurelens.errors import RepositoryError
from failurelens.indexing.base import RepositoryIndex
from failurelens.indexing.python_indexer import index_python_source


def _is_excluded(relative_posix: str, config: IndexConfig) -> bool:
    return any(fnmatch.fnmatch(relative_posix, pattern) for pattern in config.exclude_globs)


def iter_source_files(root: Path, config: IndexConfig) -> list[tuple[Path, str]]:
    """Return ``(absolute_path, repo_relative_posix)`` for each file to index."""
    exclude_dirs = set(config.exclude_dirs)
    results: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=config.follow_symlinks):
        # Prune excluded directories in-place so os.walk does not descend into them.
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)
        for filename in sorted(filenames):
            if not filename.endswith(config.include_suffixes):
                continue
            absolute = Path(dirpath) / filename
            relative = absolute.relative_to(root).as_posix()
            if _is_excluded(relative, config):
                continue
            results.append((absolute, relative))
    return results


def build_index(
    repository: str | os.PathLike[str], config: IndexConfig | None = None
) -> RepositoryIndex:
    """Build a :class:`RepositoryIndex` for ``repository``.

    Raises :class:`RepositoryError` if the path does not exist or is not a
    directory. Individual unreadable or unparseable files are recorded as
    warnings / ``parse_error`` rather than aborting the whole index.
    """
    config = config or IndexConfig()
    root = Path(repository)
    if not root.exists():
        raise RepositoryError(f"Repository path does not exist: {root}")
    if not root.is_dir():
        raise RepositoryError(f"Repository path is not a directory: {root}")

    root = root.resolve()
    index = RepositoryIndex(root=str(root))

    for absolute, relative in iter_source_files(root, config):
        try:
            size = absolute.stat().st_size
        except OSError as exc:
            index.warnings.append(f"Could not stat {relative}: {exc}")
            continue
        if config.max_file_bytes and size > config.max_file_bytes:
            index.warnings.append(
                f"Skipped large file (> {config.max_file_bytes} bytes): {relative}"
            )
            continue
        try:
            source = absolute.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            index.warnings.append(f"Could not read {relative}: {exc}")
            continue
        index.files.append(index_python_source(relative, source))

    return index
