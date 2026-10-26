"""AST-based indexer for Python source files.

Uses the standard library :mod:`ast` module (no third-party parser) to extract
classes, functions and methods with accurate line ranges. Files that fail to
parse (e.g. syntax errors) are still recorded with a ``parse_error`` so the
index remains complete and the failure is visible rather than silent.
"""

from __future__ import annotations

import ast

from failurelens.indexing.base import CodeUnit, FileIndex
from failurelens.schemas import UnitKind

LANGUAGE = "python"


class _UnitCollector(ast.NodeVisitor):
    """Walks a module AST, recording code units with qualified names."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.units: list[CodeUnit] = []
        self._scope: list[str] = []
        self._in_class: list[bool] = []

    def _qualified(self, name: str) -> str:
        return ".".join([*self._scope, name])

    def _record(self, node: ast.AST, name: str, kind: UnitKind) -> None:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            return
        self.units.append(
            CodeUnit(
                kind=kind,
                name=name,
                qualified_name=self._qualified(name),
                file=self.file_path,
                start_line=start,
                end_line=end,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, node.name, UnitKind.CLASS)
        self._scope.append(node.name)
        self._in_class.append(True)
        self.generic_visit(node)
        self._in_class.pop()
        self._scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        kind = UnitKind.METHOD if (self._in_class and self._in_class[-1]) else UnitKind.FUNCTION
        self._record(node, node.name, kind)
        self._scope.append(node.name)
        self._in_class.append(False)
        self.generic_visit(node)
        self._in_class.pop()
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)


def index_python_source(relative_path: str, source: str) -> FileIndex:
    """Index a single Python source string into a :class:`FileIndex`."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        return FileIndex(path=relative_path, language=LANGUAGE, units=[], parse_error=str(exc))

    collector = _UnitCollector(relative_path)
    collector.visit(tree)
    # A module-level unit spans the whole file and acts as a fallback location.
    line_count = source.count("\n") + 1 if source else 1
    units: list[CodeUnit] = [
        CodeUnit(
            kind=UnitKind.MODULE,
            name=relative_path.rsplit("/", 1)[-1],
            qualified_name=relative_path,
            file=relative_path,
            start_line=1,
            end_line=line_count,
        ),
        *collector.units,
    ]
    return FileIndex(path=relative_path, language=LANGUAGE, units=units)
