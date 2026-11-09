"""Tests for the stack-trace + call-graph analyzer."""

from __future__ import annotations

from pathlib import Path

import pytest

from failurelens import FailureLens
from failurelens.errors import InputError
from failurelens.inference import BASELINE_STRATEGIES, CallGraphAnalyzer
from failurelens.retrieval import callgraph
from failurelens.schemas import AnalysisStatus, EvidenceKind, ScoreType

# The failure surfaces in handle() (line 11), but the changed/broken function is
# normalize(), which handle() calls and which is NOT in the stack trace.
SERVICE_PY = '''\
"""Service."""


def normalize(value):
    return value.strip()


def validate(value):
    return normalize(value)


def handle(request):
    return validate(request).lower()


def unrelated():
    return 1
'''

ENTRY_PY = """\
from service import handle


def main():
    return handle(None)
"""

TRACEBACK = """\
Traceback (most recent call last):
  File "entry.py", line 5, in main
    return handle(None)
  File "service.py", line 13, in handle
    return validate(request).lower()
AttributeError: 'NoneType' object has no attribute 'strip'
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "service.py").write_text(SERVICE_PY, encoding="utf-8")
    (tmp_path / "entry.py").write_text(ENTRY_PY, encoding="utf-8")
    return tmp_path


def _lens() -> FailureLens:
    return FailureLens.from_baseline(strategy="stacktrace-callgraph")


def test_registered_and_labelled() -> None:
    assert BASELINE_STRATEGIES["stacktrace-callgraph"] is CallGraphAnalyzer
    result = _lens().diagnose(Path("."), error_text="no trace here")
    assert result.analysis_method == "baseline-stacktrace-callgraph"
    assert result.model_id is None


def test_ranks_stack_functions_then_call_neighbours(repo: Path) -> None:
    result = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    assert result.status is AnalysisStatus.OK
    names = [c.function for c in result.candidates]
    assert names[0] == "handle"
    assert "main" in names
    # validate() is called by handle() (one hop); normalize() is two hops away.
    assert names.index("validate") < names.index("normalize")
    assert "unrelated" not in names
    assert all(c.score_type is ScoreType.HEURISTIC for c in result.candidates)
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_evidence_states_the_relation_in_plain_words(repo: Path) -> None:
    result = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    by_name = {c.function: c for c in result.candidates}
    top = by_name["handle"]
    assert {e.kind for e in top.evidence} >= {EvidenceKind.STACK_FRAME, EvidenceKind.EXCEPTION_SITE}
    one_hop = [e for e in by_name["validate"].evidence if e.kind is EvidenceKind.CALL_RELATION]
    assert len(one_hop) == 1
    assert "One call-relation hop" in one_hop[0].description
    assert "is called by 'handle'" in one_hop[0].description
    assert one_hop[0].frame_order == 1
    two_hop = [e for e in by_name["normalize"].evidence if e.kind is EvidenceKind.CALL_RELATION]
    assert "Two call-relation hops" in two_hop[0].description


def test_output_is_deterministic(repo: Path) -> None:
    first = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    second = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    first.execution_time_ms = second.execution_time_ms = 0.0
    assert first.model_dump() == second.model_dump()


def test_top_k_truncates(repo: Path) -> None:
    result = _lens().diagnose(repo, error_text=TRACEBACK, top_k=2)
    assert [c.rank for c in result.candidates] == [1, 2]


def test_frames_outside_the_repository_give_no_candidates(repo: Path) -> None:
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/elsewhere/lib/other.py", line 3, in f\n    x()\nValueError: boom\n'
    )
    result = _lens().diagnose(repo, error_text=trace)
    assert result.status is AnalysisStatus.NO_CANDIDATES
    assert result.candidates == []


def test_module_level_frame_falls_back_to_a_file_level_candidate(repo: Path) -> None:
    trace = (
        "Traceback (most recent call last):\n"
        '  File "entry.py", line 1, in <module>\n    from service import handle\n'
        "ImportError: boom\n"
    )
    result = _lens().diagnose(repo, error_text=trace)
    assert result.status is AnalysisStatus.OK
    assert result.candidates[0].file == "entry.py"
    assert result.candidates[0].function is None
    assert result.candidates[0].start_line is None


def test_syntax_error_files_do_not_break_the_analysis(repo: Path) -> None:
    (repo / "broken.py").write_text("def oops(:\n", encoding="utf-8")
    result = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    assert result.status is AnalysisStatus.OK
    assert result.candidates[0].function == "handle"


def test_oversized_graph_is_skipped_but_the_stack_still_ranks(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(callgraph, "MAX_GRAPH_UNITS", 1)
    result = _lens().diagnose(repo, error_text=TRACEBACK, top_k=10)
    names = [c.function for c in result.candidates]
    assert names == ["handle", "main"]
    assert any("Call graph skipped" in w for w in result.warnings)


def test_recursive_functions_are_handled(tmp_path: Path) -> None:
    (tmp_path / "r.py").write_text(
        "def f(n):\n    return f(n - 1)\n\n\ndef g():\n    return f(1)\n", encoding="utf-8"
    )
    trace = (
        "Traceback (most recent call last):\n"
        '  File "r.py", line 2, in f\n    return f(n - 1)\nRecursionError: maximum depth\n'
    )
    result = _lens().diagnose(tmp_path, error_text=trace, top_k=5)
    assert [c.function for c in result.candidates] == ["f", "g"]


def test_unknown_strategy_is_an_input_error() -> None:
    with pytest.raises(InputError):
        FailureLens.from_baseline(strategy="nope")


def test_selecting_the_lexical_strategy_keeps_the_original_method(
    repo: Path,
) -> None:
    lens = FailureLens.from_baseline(strategy="stacktrace-lexical")
    result = lens.diagnose(repo, error_text=TRACEBACK)
    assert result.analysis_method == "baseline-stacktrace-lexical"
    assert result.candidates[0].function == "handle"
