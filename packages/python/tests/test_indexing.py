"""Tests for the AST indexer and repository index."""

from __future__ import annotations

from pathlib import Path

from failurelens.config import IndexConfig
from failurelens.indexing.python_indexer import index_python_source
from failurelens.indexing.repository import build_index
from failurelens.schemas import UnitKind


def test_index_python_source_extracts_units() -> None:
    source = (
        "class Greeter:\n"
        "    def hello(self):\n"
        "        return 'hi'\n"
        "\n"
        "def top_level():\n"
        "    return 1\n"
    )
    file_index = index_python_source("mod.py", source)
    assert file_index.parse_error is None
    kinds = {unit.qualified_name: unit.kind for unit in file_index.units}
    assert kinds["Greeter"] is UnitKind.CLASS
    assert kinds["Greeter.hello"] is UnitKind.METHOD
    assert kinds["top_level"] is UnitKind.FUNCTION
    # A module-level unit is always recorded.
    assert any(unit.kind is UnitKind.MODULE for unit in file_index.units)


def test_index_records_parse_error_without_raising() -> None:
    file_index = index_python_source("broken.py", "def oops(:\n")
    assert file_index.parse_error is not None
    # The file is still represented (with no units) so indexing never crashes.
    assert file_index.units == []


def test_unit_at_returns_tightest_enclosing_unit() -> None:
    source = "class A:\n    def method(self):\n        x = 1\n        return x\n"
    file_index = index_python_source("a.py", source)
    unit = file_index.unit_at(3)
    assert unit is not None
    assert unit.qualified_name == "A.method"


def test_build_index_scans_repo_and_computes_counts(sample_repo: Path) -> None:
    index = build_index(sample_repo)
    paths = set(index.paths())
    assert "app.py" in paths
    assert "pkg/core.py" in paths
    assert "pkg/__init__.py" in paths
    assert index.file_count() == 3
    assert index.unit_count() > 0
    core = index.file("pkg/core.py")
    assert core is not None
    assert {u.name for u in core.units if u.kind is not UnitKind.MODULE} == {"divide", "average"}


def test_build_index_excludes_configured_directories(sample_repo: Path) -> None:
    (sample_repo / ".venv").mkdir()
    (sample_repo / ".venv" / "ignored.py").write_text("x = 1\n", encoding="utf-8")
    index = build_index(sample_repo, IndexConfig())
    assert all(".venv" not in path for path in index.paths())
