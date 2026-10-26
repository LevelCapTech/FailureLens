"""Tests for the baseline ranking heuristic."""

from __future__ import annotations

from failurelens.ranking.baseline import extract_error_tokens, rank_candidates
from failurelens.retrieval.baseline import RawCandidate
from failurelens.schemas import ScoreType


def _candidate(
    file: str,
    *,
    function: str | None = None,
    max_frame_order: int = 0,
    is_exception_site: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
) -> RawCandidate:
    return RawCandidate(
        file=file,
        function=function,
        qualified_name=function,
        start_line=start_line,
        end_line=end_line,
        max_frame_order=max_frame_order,
        is_exception_site=is_exception_site,
    )


def test_deeper_exception_site_ranks_first() -> None:
    shallow = _candidate("a.py", function="outer", max_frame_order=0)
    deep = _candidate("b.py", function="inner", max_frame_order=1, is_exception_site=True)
    ranked = rank_candidates(
        [shallow, deep], frame_count=2, error_message="boom", source_reader=None
    )
    assert [c.function for c in ranked] == ["inner", "outer"]
    assert ranked[0].rank == 1
    assert ranked[0].score > ranked[1].score
    assert ranked[0].score_type is ScoreType.HEURISTIC


def test_lexical_match_increases_score() -> None:
    candidate = _candidate("a.py", function="f", max_frame_order=0, start_line=1, end_line=1)
    message = "name 'count' is not defined"

    without = rank_candidates([candidate], frame_count=1, error_message=message, source_reader=None)
    with_source = rank_candidates(
        [candidate],
        frame_count=1,
        error_message=message,
        source_reader=lambda _file: "count = 0\n",
    )
    assert with_source[0].score > without[0].score
    # The matched token is surfaced as evidence.
    assert any("count" in ev.description for ev in with_source[0].evidence)


def test_ranking_is_deterministic_on_ties() -> None:
    # Identical scores must tie-break deterministically (frame order, then path).
    a = _candidate("z.py", function="a", max_frame_order=0)
    b = _candidate("a.py", function="b", max_frame_order=0)
    first = rank_candidates([a, b], frame_count=1, error_message=None)
    second = rank_candidates([b, a], frame_count=1, error_message=None)
    assert [c.file for c in first] == [c.file for c in second]


def test_top_k_truncates_results() -> None:
    candidates = [_candidate(f"f{i}.py", max_frame_order=i) for i in range(5)]
    ranked = rank_candidates(candidates, frame_count=5, error_message=None, top_k=2)
    assert len(ranked) == 2
    assert [c.rank for c in ranked] == [1, 2]


def test_extract_error_tokens_filters_stopwords_and_short_words() -> None:
    tokens = extract_error_tokens("name 'count' is not defined")
    assert "count" in tokens
    assert "defined" in tokens
    assert "is" not in tokens  # too short
    assert "not" not in tokens  # stopword


def test_extract_error_tokens_handles_empty() -> None:
    assert extract_error_tokens(None) == set()
    assert extract_error_tokens("") == set()
