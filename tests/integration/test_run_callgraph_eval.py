"""Tests for the v0.5 before/after driver's pure helpers (synthetic data only)."""

from __future__ import annotations

from evaluation.run_callgraph_eval import candidate_id, extend_with_pool, rule_checks

from failurelens.schemas import Candidate, ScoreType


def _candidate(**kw: object) -> Candidate:
    base: dict[str, object] = {"rank": 1, "file": "a.py", "score": 0.5}
    base.update(kw)
    return Candidate(score_type=ScoreType.HEURISTIC, **base)  # type: ignore[arg-type]


def test_candidate_id_matches_pool_format_and_skips_file_level() -> None:
    c = _candidate(function="f", start_line=3, end_line=9)
    assert candidate_id(c) == "a.py::f:3-9"
    assert candidate_id(_candidate()) is None


def test_extend_with_pool_keeps_library_order_then_ids_ascending() -> None:
    pool = ["a::x:1-2", "b::y:1-2", "c::z:1-2", "d::w:1-2"]
    ordered = extend_with_pool(["c::z:1-2", None, "zzz::not-in-pool:1-2", "a::x:1-2"], pool)
    assert ordered == ("c::z:1-2", "a::x:1-2", "b::y:1-2", "d::w:1-2")


def test_rule_checks_applies_the_adoption_rule_literally() -> None:
    strata = ["r1"] * 5 + ["r2"] * 5 + ["r3"] * 5
    before = ([0.1] * 15, [0.0] * 15)
    after = ([0.5] * 15, [0.4] * 15)
    checks = rule_checks(strata, before, after, p95_seconds=0.2, rss_gib=0.1)
    assert checks["criterion_1_mrr_gain_ge_0.10_and_ci_lower_gt_0"]["pass"] is True
    assert checks["criterion_2_leave_one_repo_out_all_positive"]["pass"] is True
    assert checks["criterion_4_speed_and_memory"]["pass"] is True

    small = rule_checks(strata, before, ([0.15] * 15, [0.0] * 15), 0.2, 0.1)
    assert small["criterion_1_mrr_gain_ge_0.10_and_ci_lower_gt_0"]["pass"] is False

    slow = rule_checks(strata, before, after, p95_seconds=6.0, rss_gib=0.1)
    assert slow["criterion_4_speed_and_memory"]["pass"] is False
