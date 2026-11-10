"""Synthetic-only tests for the v0.7 group-T features, rule candidates and selection."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from evaluation import run_v07 as driver
from evaluation import v07_features as v07
from evaluation.private_adapter import Candidate, RankRequest

from failurelens import FailureLens

LOG = (
    "Traceback (most recent call last):\n"
    '  File "tests/test_mod.py", line 3, in test_it\n'
    "    return core(1)\n"
    '  File "pkg/mod.py", line 3, in core\n'
    "    return helper(x)\n"
    "KeyError: 'widget'\n"
)


def _request() -> RankRequest:
    rows = {
        "pkg/mod.py::core:1-4": "def core(x):\n    y = 1\n    return helper(x)\n",
        "pkg/mod.py::helper:6-8": "def helper(x):\n    return x\n",
        "pkg/mod.py::widget_lookup:10-12": "def widget_lookup(self):\n    return 1\n",
        "tests/test_mod.py::test_it:1-4": "def test_it():\n    x = 0\n    return core(1)\n",
        "tests/test_mod.py::test_other:6-8": "def test_other():\n    return 0\n",
    }
    candidates = tuple(
        Candidate(
            identifier,
            identifier.split("::", 1)[0],
            identifier.rsplit(":", 1)[0],
            source,
            hashlib.sha256(source.encode()).hexdigest(),
        )
        for identifier, source in rows.items()
    )
    query = json.dumps({"error_log": LOG, "stack_trace": LOG})
    return RankRequest(query, candidates, 5, "digest")


def test_group_t_features_are_deterministic_and_correct() -> None:
    parse = FailureLens.from_baseline().parsers.parse
    request = _request()
    first = v07.case_v07(request, parse)
    second = v07.case_v07(request, parse)
    assert np.array_equal(first.t, second.t)
    ids = list(first.ids)
    col = {name: i for i, name in enumerate(v07.T_NAMES)}
    core = ids.index("pkg/mod.py::core:1-4")
    helper = ids.index("pkg/mod.py::helper:6-8")
    widget = ids.index("pkg/mod.py::widget_lookup:10-12")
    test_it = ids.index("tests/test_mod.py::test_it:1-4")
    assert first.t[test_it, col["t01_is_test_file"]] == 1.0
    assert first.t[core, col["t01_is_test_file"]] == 0.0
    assert first.t[core, col["t02_production_stack"]] == 1.0
    assert first.t[test_it, col["t02_production_stack"]] == 0.0
    assert first.t[helper, col["t03_production_hops"]] == 1 / 3
    assert first.t[widget, col["t06_name_in_message"]] == 0.5
    assert first.t[widget, col["t07_same_class_near"]] == 0.0  # core is not a method


def test_rule_a_puts_test_files_last() -> None:
    parse = FailureLens.from_baseline().parsers.parse
    case = v07.case_v07(_request(), parse)
    scores = list(v07.rule_scores(case, "A"))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    flags = [bool(case.is_test[i]) for i in order]
    assert flags == sorted(flags)  # production first, then test files


def test_selection_prefers_top1_then_mrr_then_rule_candidates() -> None:
    metrics = {
        "A": ([0.5, 0.5], [1.0, 0.0]),
        "D": ([0.6, 0.6], [1.0, 0.0]),
        "F": ([0.9, 0.9], [0.0, 0.0]),
    }
    assert driver.select(metrics, ["A", "D", "F"]) == "D"
    tied = {"A": ([0.5, 0.5], [1.0, 0.0]), "D": ([0.5, 0.5], [1.0, 0.0])}
    assert driver.select(tied, ["A", "D"]) == "A"


def _rules(best: tuple[list[float], list[float]], p95: float = 1.0) -> dict[str, object]:
    strata = ["r1"] * 10 + ["r2"] * 10
    v5 = ([0.4] * 20, [0.2] * 20)
    return driver.apply_rules(
        best=best,
        v5=v5,
        strata=strata,
        loro_mrr=[0.6] * 20,
        cv_mrr=0.5,
        v5_train_mrr=0.4,
        p95_seconds=p95,
        peak_rss_gib=1.0,
    )


def test_rules_require_top1_gain_and_speed() -> None:
    good = ([0.6] * 20, [0.4] * 20)
    assert _rules(good)["all_pass"] is True
    flat_top1 = ([0.6] * 20, [0.2] * 20)
    assert _rules(flat_top1)["R1_gain_over_v5"]["pass"] is False  # type: ignore[index]
    assert _rules(good, p95=6.0)["R5_speed_memory_offline"]["pass"] is False  # type: ignore[index]
