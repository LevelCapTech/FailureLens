"""Core data structures for a repository index.

The index is a plain, serializable description of the code units (modules,
classes, functions, methods) found in a repository, with their file locations
and line ranges. It is deliberately model/engine-agnostic so it can be cached
and reused across analyses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from failurelens.schemas import UnitKind


class CodeUnit(BaseModel):
    """A single addressable unit of code within a file."""

    model_config = ConfigDict(extra="forbid")

    kind: UnitKind
    name: str
    qualified_name: str = Field(description="Dotted name including enclosing scopes.")
    file: str = Field(description="Repo-relative POSIX path.")
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    def contains(self, line: int) -> bool:
        """Return ``True`` if ``line`` falls within this unit's span (inclusive)."""
        return self.start_line <= line <= self.end_line

    def span(self) -> int:
        return self.end_line - self.start_line + 1


class FileIndex(BaseModel):
    """Indexing result for one source file."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Repo-relative POSIX path.")
    language: str
    units: list[CodeUnit] = Field(default_factory=list)
    parse_error: str | None = Field(
        default=None,
        description="Set when the file could not be parsed (indexing still records it).",
    )

    def unit_at(self, line: int) -> CodeUnit | None:
        """Return the smallest unit containing ``line``, or ``None``.

        When units are nested (e.g. a method inside a class), the tightest
        enclosing unit wins so callers get the most specific location.
        """
        best: CodeUnit | None = None
        for unit in self.units:
            if unit.kind is UnitKind.MODULE:
                continue
            if unit.contains(line) and (best is None or unit.span() < best.span()):
                best = unit
        return best


class RepositoryIndex(BaseModel):
    """An index over all enumerated files in a repository."""

    model_config = ConfigDict(extra="forbid")

    root: str = Field(description="Absolute path to the repository root.")
    schema_version: str = Field(default="0.1.0")
    files: list[FileIndex] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    def file(self, path: str) -> FileIndex | None:
        for file_index in self.files:
            if file_index.path == path:
                return file_index
        return None

    def file_count(self) -> int:
        return len(self.files)

    def unit_count(self) -> int:
        return sum(len(f.units) for f in self.files)
