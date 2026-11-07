"""Synthetic-only tests for the v0.4 full-pool loader, scoring components and driver logic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from evaluation import full_pool_methods as methods
from evaluation import run_full_pool as driver
from evaluation.private_adapter import Candidate, PrivateInputError
from evaluation.product_pool import load_product_cases
from test_product_pool import _fixture

from failurelens import FailureLens


def _load_full(paths):
    records, source, train, trees, _ = paths
    return load_product_cases(
        records,
        hashlib.sha256(records.read_bytes()).hexdigest(),
        source,
        train,
        trees,
        top_k=2,
        production=False,
        allow_ambiguous_retrieved_names=True,
        pool="full",
    )


def test_full_pool_contains_every_function_and_shipped_order_first(tmp_path: Path) -> None:
    cases, baseline = _load_full(_fixture(tmp_path))
    ids = [c.id for c in cases[0].request.candidates]
    assert ids == ["mod.py::cause:1-2", "mod.py::decoy:4-5"]
    assert cases[0].positive_function_ids == {"mod.py::cause"}
    order = baseline(cases[0].request).ordered_ids
    assert sorted(order) == sorted(ids)
    assert order[0] == "mod.py::decoy:4-5"  # deepest frame first in the shipped order


def test_invalid_pool_mode_is_rejected(tmp_path: Path) -> None:
    records, source, train, trees, _ = _fixture(tmp_path)
    with pytest.raises(PrivateInputError):
        load_product_cases(
            records, "0" * 64, source, train, trees, top_k=2, production=False, pool="all"
        )


def test_tokenize_splits_identifiers_and_drops_noise() -> None:
    assert methods.tokenize("parseHTTPResponse get_user_name for x") == [
        "parse",
        "http",
        "response",
        "get",
        "user",
        "name",
    ]


def _candidate(name: str, source: str, start: int = 1, end: int = 3) -> Candidate:
    identifier = f"m.py::{name}:{start}-{end}"
    return Candidate(
        identifier, "m.py", f"m.py::{name}", source, hashlib.sha256(source.encode()).hexdigest()
    )


def test_bm25_prefers_rare_matching_terms_and_is_deterministic() -> None:
    pool = [
        _candidate("alpha", "def alpha():\n    return widget_count\n"),
        _candidate("beta", "def beta():\n    return other\n"),
        _candidate("gamma", "def gamma():\n    return other\n"),
    ]
    scores = methods.bm25_scores("KeyError: widget_count missing", pool)
    assert scores[0] > 0 and scores[1] == scores[2] == 0
    assert scores == methods.bm25_scores("KeyError: widget_count missing", pool)
    assert methods.bm25_scores("", pool) == [0.0, 0.0, 0.0]


def test_stack_and_graph_signals_follow_frames_and_calls() -> None:
    pool = [
        _candidate("top", "def top():\n    return helper()\n", 1, 2),
        _candidate("helper", "def helper():\n    return leaf()\n", 4, 5),
        _candidate("leaf", "def leaf():\n    return 1\n", 7, 8),
        _candidate("far", "def far():\n    return 0\n", 10, 11),
    ]
    log = 'Traceback (most recent call last):\n  File "m.py", line 2, in top\nValueError: x'
    lens = FailureLens.from_baseline()
    stack, seeds = methods.stack_signal(log, pool, lens.parsers.parse)
    assert seeds == {0} and stack[0] == pytest.approx(1.0) and stack[1:] == [0.0, 0.0, 0.0]
    graph = methods.graph_signal(methods.call_graph(pool), seeds)
    assert graph == [0.0, 0.5, 0.25, 0.0]


def test_fusion_and_ordering_ties_break_by_id() -> None:
    components = methods.CaseComponents(
        ("b", "a", "c"), (0.0, 0.0, 1.0), (0, 0, 0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )
    similarity = [0.5, 0.5, 0.1]
    assert methods.order_by(
        components.ids, methods.fused_scores(components, similarity, (1, 0, 0, 0))
    ) == (
        "a",
        "b",
        "c",
    )
    assert (
        methods.order_by(
            components.ids, methods.fused_scores(components, similarity, (0, 1, 0, 0))
        )[0]
        == "c"
    )


def test_selection_ties_and_stop_rule() -> None:
    metrics = {
        "L1": ([0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]),
        "E-x": ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
        "F-x|1,0,0,0": ([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
        "F-x|1,1,0,0": ([0.2, 0.2, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]),
    }
    best = driver.select_best(metrics, ["F-x|1,1,0,0", "F-x|1,0,0,0", "E-x"])
    assert best == "E-x"  # ties: fewer non-zero weights, then smaller key
    strata = ["r1", "r1", "r2", "r2"]
    assert driver.stop_rule(metrics[best][0], metrics["L1"][0], strata)["pass"] is True
    weak = [0.52, 0.52, 0.52, 0.52]
    assert driver.stop_rule(weak, metrics["L1"][0], strata)["pass"] is False
    loro = driver.leave_one_repo_out(strata, metrics, ["E-x", "F-x|1,1,0,0"])
    assert loro == [1.0, 1.0, 1.0, 1.0]


def test_grid_is_the_preregistered_one() -> None:
    assert len(driver.GRID) == 80 and (0, 0, 0, 0) not in driver.GRID


def test_experiment_on_synthetic_case_and_report_permissions(tmp_path: Path) -> None:
    cases, baseline = _load_full(_fixture(tmp_path))
    lens = FailureLens.from_baseline()
    components = [methods.build_components(c.request, lens.parsers.parse) for c in cases]
    sims = {"fake": [[0.9, 0.1]]}
    report = driver.run_experiment(cases, baseline, components, sims)
    assert report["cases"] == 1 and report["pool_recall"]["function_recall"] == 1.0
    assert report["configurations_searched"] == 81
    path = driver.write_report(report, tmp_path / "out", "r.json")
    assert path.stat().st_mode & 0o777 == 0o600
    json.loads(path.read_text())  # serializable
