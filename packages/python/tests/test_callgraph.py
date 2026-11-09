"""Tests for the static, name-based call graph."""

from __future__ import annotations

import pytest

from failurelens.indexing.base import CodeUnit
from failurelens.retrieval import callgraph
from failurelens.retrieval.callgraph import build_call_graph, called_names, reach_from
from failurelens.schemas import UnitKind


def _unit(name: str, start: int = 1) -> CodeUnit:
    return CodeUnit(
        kind=UnitKind.FUNCTION,
        name=name,
        qualified_name=name,
        file="m.py",
        start_line=start,
        end_line=start + 1,
    )


def _graph(sources: dict[str, str]):  # type: ignore[no-untyped-def]
    units = [_unit(name, start=1 + 10 * i) for i, name in enumerate(sources)]
    by_unit = {id(u): s for u, s in zip(units, sources.values(), strict=True)}
    return build_call_graph(units, lambda unit: by_unit[id(unit)])


def test_called_names_collects_names_and_attributes() -> None:
    source = "def f():\n    a()\n    obj.b()\n    x = c\n    return d.e.f()\n"
    assert called_names(source) == {"a", "b", "f"}


def test_called_names_dedents_methods() -> None:
    assert called_names("    def m(self):\n        helper()\n") == {"helper"}


@pytest.mark.parametrize("source", ["def broken(:\n", "    def x(:\n  y", "\x00"])
def test_called_names_never_raises_on_unparsable_text(source: str) -> None:
    assert called_names(source) == set()


def test_edges_are_undirected_but_calls_are_directed() -> None:
    graph = _graph({"a": "def a():\n    b()\n", "b": "def b():\n    pass\n"})
    assert graph.calls[0] == {1}
    assert graph.calls[1] == set()
    assert graph.adjacency[0] == {1}
    assert graph.adjacency[1] == {0}


def test_recursion_and_cycles_do_not_loop() -> None:
    graph = _graph(
        {
            "a": "def a():\n    a()\n    b()\n",
            "b": "def b():\n    a()\n",
        }
    )
    assert 0 not in graph.adjacency[0]
    reached = reach_from(graph, [0])
    assert reached[0].hops == 0
    assert reached[1].hops == 1


def test_names_shared_by_many_units_are_ignored() -> None:
    sources = {"caller": "def caller():\n    run()\n"}
    for i in range(callgraph.MAX_SAME_NAME + 1):
        sources[f"run{i}"] = "def run():\n    pass\n"
    units = [_unit("caller")] + [_unit("run", start=10 + 10 * i) for i in range(9)]
    texts = [sources["caller"]] + ["def run():\n    pass\n"] * 9
    graph = build_call_graph(units, lambda unit: texts[units.index(unit)])
    assert graph.adjacency[0] == set()


def test_reach_is_limited_to_two_hops_and_reports_origin() -> None:
    graph = _graph(
        {
            "a": "def a():\n    b()\n",
            "b": "def b():\n    c()\n",
            "c": "def c():\n    d()\n",
            "d": "def d():\n    pass\n",
        }
    )
    reached = reach_from(graph, [0])
    assert {i: r.hops for i, r in reached.items()} == {0: 0, 1: 1, 2: 2}
    assert all(r.origin == 0 for r in reached.values())


def test_reach_is_deterministic_and_prefers_lowest_seed() -> None:
    graph = _graph(
        {
            "a": "def a():\n    mid()\n",
            "b": "def b():\n    mid()\n",
            "mid": "def mid():\n    pass\n",
        }
    )
    first = reach_from(graph, [1, 0])
    second = reach_from(graph, [0, 1])
    assert first == second
    assert first[2].origin == 0


def test_unreadable_or_oversized_units_are_skipped_with_a_message() -> None:
    units = [_unit("a"), _unit("b", start=10)]
    texts = {"a": None, "b": "x" * (callgraph.MAX_UNIT_SOURCE_CHARS + 1)}
    graph = build_call_graph(units, lambda unit: texts[unit.name])
    assert graph.skipped_units == 2
    assert graph.skipped_reason is not None and "incomplete" in graph.skipped_reason


def test_graph_is_skipped_beyond_the_unit_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(callgraph, "MAX_GRAPH_UNITS", 1)
    graph = build_call_graph([_unit("a"), _unit("b", start=10)], lambda unit: "def f():\n    g()\n")
    assert graph.skipped_reason is not None and "skipped" in graph.skipped_reason
    assert all(not edges for edges in graph.adjacency)
