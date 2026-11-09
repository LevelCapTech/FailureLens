"""Static, name-based call graph over the functions and methods of a repository.

This is deliberately simple and over-approximate: a function is linked to every
function or method that has the name it calls (``foo()`` or ``obj.foo()``), in
either direction. Dynamic dispatch, aliasing and imports are not resolved. Names
shared by more than :data:`MAX_SAME_NAME` units are ignored because they carry
almost no localizing signal. Pure standard library, deterministic, offline.

The definition (and the decay weights) follow ``docs/releases/v0.4/selection.md``
and are intentionally not re-tuned.
"""

from __future__ import annotations

import ast
import textwrap
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from failurelens.indexing.base import CodeUnit

#: Names that match more units than this are ignored when linking calls.
MAX_SAME_NAME = 8
#: Distance (in call-relation hops) -> signal. Seeds themselves get no graph signal.
HOP_DECAY: dict[int, float] = {1: 0.5, 2: 0.25}
MAX_HOPS = 2
#: Cost bounds for large repositories. Beyond them the graph is skipped, not guessed.
MAX_GRAPH_UNITS = 50_000
MAX_UNIT_SOURCE_CHARS = 100_000

UnitSource = Callable[[CodeUnit], str | None]


def called_names(source: str) -> set[str]:
    """Names called in ``source`` (``foo()`` -> ``foo``, ``a.foo()`` -> ``foo``).

    Returns an empty set when the text does not parse (never raises).
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except (SyntaxError, ValueError, RecursionError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


@dataclass
class CallGraph:
    """Adjacency over unit indices. ``calls[i]`` holds the units whose name ``i`` calls."""

    units: Sequence[CodeUnit]
    calls: list[set[int]]
    adjacency: list[set[int]]
    skipped_units: int = 0
    skipped_reason: str | None = None
    #: Unit-name -> indices, kept so callers can explain relations.
    names: dict[str, list[int]] = field(default_factory=dict)


def build_call_graph(units: Sequence[CodeUnit], source_of: UnitSource) -> CallGraph:
    """Link ``units`` by the names they call. ``source_of`` returns a unit's source span."""
    empty = CallGraph(units, [set() for _ in units], [set() for _ in units])
    if len(units) > MAX_GRAPH_UNITS:
        empty.skipped_units = len(units)
        empty.skipped_reason = (
            f"Call graph skipped: {len(units)} code units exceed the limit of {MAX_GRAPH_UNITS}."
        )
        return empty
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        by_name[unit.name].append(index)
    calls: list[set[int]] = [set() for _ in units]
    adjacency: list[set[int]] = [set() for _ in units]
    skipped = 0
    for index, unit in enumerate(units):
        source = source_of(unit)
        if source is None or len(source) > MAX_UNIT_SOURCE_CHARS:
            skipped += 1
            continue
        for name in called_names(source):
            targets = by_name.get(name, [])
            if not targets or len(targets) > MAX_SAME_NAME:
                continue
            for target in targets:
                if target != index:
                    calls[index].add(target)
                    adjacency[index].add(target)
                    adjacency[target].add(index)
    return CallGraph(
        units,
        calls,
        adjacency,
        skipped_units=skipped,
        skipped_reason=(
            f"Call graph incomplete: {skipped} code unit(s) were too large or unreadable."
            if skipped
            else None
        ),
        names=dict(by_name),
    )


@dataclass(frozen=True)
class Reach:
    """How a unit was reached from the stack: hops and the stack unit it came from."""

    hops: int
    origin: int


def reach_from(graph: CallGraph, seeds: Sequence[int]) -> dict[int, Reach]:
    """Breadth-first distances (<= :data:`MAX_HOPS`) from the stack units, deterministic.

    Seeds map to ``Reach(0, itself)``. Among equally near seeds the lowest unit
    index is the origin, so explanations are stable.
    """
    reached: dict[int, Reach] = {}
    queue: deque[int] = deque()
    for seed in sorted(seeds):
        reached[seed] = Reach(0, seed)
        queue.append(seed)
    while queue:
        node = queue.popleft()
        here = reached[node]
        if here.hops >= MAX_HOPS:
            continue
        for neighbour in sorted(graph.adjacency[node]):
            if neighbour not in reached:
                reached[neighbour] = Reach(here.hops + 1, here.origin)
                queue.append(neighbour)
    return reached
