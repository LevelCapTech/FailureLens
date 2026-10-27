"""Tests for baseline candidate retrieval (frame resolution + candidate build)."""

from __future__ import annotations

from pathlib import Path

from failurelens.indexing.repository import build_index
from failurelens.parsers.python_traceback import PythonTracebackParser
from failurelens.retrieval.baseline import (
    resolve_frame_path,
    resolve_frames,
    retrieve_candidates,
)


def test_resolve_frame_path_matches_longest_suffix(sample_repo: Path) -> None:
    index = build_index(sample_repo)
    # Absolute path from "another machine" still resolves by path suffix.
    resolved = resolve_frame_path("/elsewhere/checkout/pkg/core.py", index)
    assert resolved == "pkg/core.py"


def test_resolve_frame_path_falls_back_to_unique_basename(sample_repo: Path) -> None:
    index = build_index(sample_repo)
    assert resolve_frame_path("/weird/location/app.py", index) == "app.py"


def test_resolve_frame_path_returns_none_when_unknown(sample_repo: Path) -> None:
    index = build_index(sample_repo)
    assert resolve_frame_path("totally/unrelated.py", index) is None
    assert resolve_frame_path(None, index) is None


def test_retrieve_candidates_marks_exception_site(sample_repo: Path, sample_traceback: str) -> None:
    index = build_index(sample_repo)
    parsed = PythonTracebackParser().parse(sample_traceback)
    assert parsed is not None
    frames = resolve_frames(parsed.frames, index)
    assert all(f.in_repository for f in frames)

    candidates = retrieve_candidates(frames, index)
    by_function = {c.function: c for c in candidates}
    assert "divide" in by_function
    assert "average" in by_function
    # The deepest in-repository frame (divide) is the exception site.
    assert by_function["divide"].is_exception_site is True
    assert by_function["average"].is_exception_site is False


def test_retrieve_candidates_empty_when_no_repo_frames(sample_repo: Path) -> None:
    index = build_index(sample_repo)
    text = (
        "Traceback (most recent call last):\n"
        '  File "/usr/lib/python3.11/json/decoder.py", line 1, in decode\n'
        "    raise ValueError\n"
        "ValueError: bad\n"
    )
    parsed = PythonTracebackParser().parse(text)
    assert parsed is not None
    frames = resolve_frames(parsed.frames, index)
    assert retrieve_candidates(frames, index) == []
