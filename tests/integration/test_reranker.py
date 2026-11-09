"""Synthetic-only tests for the v0.6 reranker features, candidates, selection rules and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from evaluation import full_pool_methods as methods
from evaluation import reranker_features as features
from evaluation import reranker_models as models
from evaluation import run_reranker as driver
from evaluation.private_adapter import Candidate, RankRequest
from test_full_pool import _load_full
from test_product_pool import _fixture

from failurelens import FailureLens

LOG = (
    "Traceback (most recent call last):\n"
    '  File "pkg/entry.py", line 3, in top\n'
    "    value = middle(x)\n"
    "ValueError: bad value in helper\n"
)


def _request(sources: dict[str, tuple[int, int, str]], log: str = LOG) -> RankRequest:
    """Candidates from ``{id_name: (start, end, source)}`` in a single file ``pkg/entry.py``."""
    candidates = []
    for name, (start, end, source) in sources.items():
        identifier = f"pkg/entry.py::{name}:{start}-{end}"
        candidates.append(
            Candidate(
                identifier,
                "pkg/entry.py",
                f"pkg/entry.py::{name}",
                source,
                hashlib.sha256(source.encode()).hexdigest(),
            )
        )
    query = json.dumps({"error_log": log, "stack_trace": log})
    return RankRequest(query, tuple(candidates), 5, "digest")


def _graph_request() -> RankRequest:
    return _request(
        {
            "top": (1, 4, "def top(x):\n    value = middle(x)\n    return value\n"),
            "middle": (6, 8, "def middle(x):\n    return helper(x) + 1\n"),
            "helper": (10, 12, "def helper(x):\n    return x\n"),
            "caller": (14, 16, "def caller():\n    return top(1)\n"),
            "unrelated": (18, 20, "def unrelated():\n    return 0\n"),
        }
    )


def _index(name: str, request: RankRequest) -> int:
    return [c.id.split("::", 1)[1].rsplit(":", 1)[0] for c in request.candidates].index(name)


def test_feature_matrix_shape_names_and_determinism() -> None:
    lens = FailureLens.from_baseline()
    request = _graph_request()
    first = features.case_features(request, lens.parsers.parse)
    second = features.case_features(request, lens.parsers.parse)
    assert first.g1.shape == (5, len(features.G1_NAMES)) == (5, 23)
    assert np.array_equal(first.g1, second.g1)
    assert np.isfinite(first.g1).all()


def test_stack_and_directed_call_features() -> None:
    lens = FailureLens.from_baseline()
    request = _graph_request()
    got = features.case_features(request, lens.parsers.parse).g1
    col = {name: i for i, name in enumerate(features.G1_NAMES)}
    top, middle, helper, caller, unrelated = (
        _index(n, request) for n in ("top", "middle", "helper", "caller", "unrelated")
    )
    assert got[top, col["f02_in_stack"]] == 1.0
    assert got[top, col["f03_deepest_frame"]] == 1.0
    assert got[top, col["f08_min_hops"]] == 0.0
    # middle is called by the stack function; caller calls the stack function.
    assert got[middle, col["f05_called_by_stack"]] == 1.0
    assert got[middle, col["f06_calls_stack"]] == 0.0
    assert got[caller, col["f06_calls_stack"]] == 1.0
    assert got[caller, col["f05_called_by_stack"]] == 0.0
    assert got[middle, col["f08_min_hops"]] == 1.0
    # helper is two calls away from the stack function.
    assert got[helper, col["f07_two_hops"]] == 1.0
    assert got[helper, col["f08_min_hops"]] == 2.0
    assert got[unrelated, col["f08_min_hops"]] == float(features.MAX_HOPS)
    assert got[unrelated, col["f02_in_stack"]] == 0.0
    # the log names "helper" and the error type's value text
    assert got[helper, col["f15_name_in_log"]] == 1.0
    assert got[top, col["f23_stack_found"]] == 1.0


def test_directed_graph_agrees_with_the_v04_undirected_graph() -> None:
    request = _graph_request()
    callees, callers = features.directed_call_graph(request.candidates)
    undirected = methods.call_graph(request.candidates)
    assert [callees[i] | callers[i] for i in range(len(callees))] == undirected


def test_stack_resolution_matches_the_v04_stack_signal(tmp_path: Path) -> None:
    lens = FailureLens.from_baseline()
    cases, _ = _load_full(_fixture(tmp_path))
    request = cases[0].request
    log = methods.error_log(request)
    _, seeds = methods.stack_signal(log, request.candidates, lens.parsers.parse)
    info = features.stack_info(log, request.candidates, lens.parsers.parse)
    assert set(info.deepest_order_by_candidate) == seeds
    got = features.case_features(request, lens.parsers.parse)
    assert got.seeds == frozenset(seeds)
    assert got.g1[:, 1].tolist() == [0.0, 1.0] or got.g1[:, 1].sum() == len(seeds)


def test_rank_inverse_breaks_ties_by_id() -> None:
    assert features._rank_inverse([1.0, 1.0, 0.0], ["b", "a", "c"]) == [0.5, 1.0, 1 / 3]
    matrix = features.embedding_features([0.2, 0.8, 0.5], ["a", "b", "c"])
    assert matrix.shape == (3, 2)
    assert matrix[1, 0] == 1.0 and matrix[0, 0] == 0.0
    assert matrix[1, 1] == 1.0


def _case(index: int, signal: bool = True, size: int = 40) -> models.TrainCase:
    rng = np.random.RandomState(index)
    g1 = rng.rand(size, len(features.G1_NAMES))
    g1[:, models.COL_IN_STACK] = 0.0  # no stack function and no graph neighbour by default
    g1[:, models.COL_MIN_HOPS] = 3.0
    g2 = rng.rand(size, 2)
    positive = int(rng.randint(size))
    if signal:
        g1[positive, 0] = 5.0  # a clear stack signal
        g1[positive, 9] = 5.0
        g2[positive, 0] = 5.0
    ids = tuple(f"f.py::fn{i:03d}:{i}-{i + 1}" for i in range(size))
    return models.TrainCase(g1, g2, (positive,), ids, index)


def test_training_subset_is_seeded_and_keeps_positives() -> None:
    case = _case(3, size=500)
    first, second = models.training_subset(case), models.training_subset(case)
    assert np.array_equal(first, second)
    assert case.positives[0] in first
    assert len(first) <= 500
    other = models.training_subset(_case(4, size=500))
    assert not np.array_equal(first, other)


@pytest.mark.parametrize("config", models.CONFIGS, ids=lambda c: c.key)
def test_every_candidate_learns_a_planted_signal(config: models.RerankerConfig) -> None:
    pytest.importorskip("sklearn")
    train = [_case(i) for i in range(30)]
    model = models.fit_reranker(config, train)
    hits = 0
    for case in [_case(100 + i) for i in range(20)]:
        matrix = models.design_matrix(config, case.g1, case.g2)
        hits += int(np.argmax(model.score(matrix)) == case.positives[0])
    assert hits >= 18
    repeat = models.fit_reranker(config, train)
    probe = models.design_matrix(config, train[0].g1, train[0].g2)
    assert np.allclose(model.score(probe), repeat.score(probe))
    coefficients = model.coefficients()
    if config.family == "boosting":
        assert coefficients is None
    else:
        assert coefficients is not None and len(coefficients) == probe.shape[1]


def test_design_matrix_requires_embedding_group_when_configured() -> None:
    embed = next(c for c in models.CONFIGS if c.uses_embedding)
    with pytest.raises(ValueError):
        models.design_matrix(embed, np.zeros((2, 23)), None)


def test_artifact_roundtrip_and_digest_check(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    config = models.CONFIGS[0]
    model = models.fit_reranker(config, [_case(i) for i in range(10)])
    path = tmp_path / "m" / "model.pkl"
    digest = models.save_artifact(model, path)
    assert (path.stat().st_mode & 0o777) == 0o600
    loaded = models.load_artifact(path, digest)
    probe = _case(50).g1
    assert np.allclose(model.score(probe), loaded.score(probe))
    with pytest.raises(ValueError):
        models.load_artifact(path, "0" * 64)


def _metric(values: list[float]) -> driver.CaseMetric:
    return values, values


def test_select_candidate_tie_breaks() -> None:
    metrics = {"A": _metric([1.0, 0.0]), "B": _metric([0.5, 0.5]), "F": _metric([0.5, 0.5])}
    flags = {"A": False, "B": False, "F": True}
    assert driver.select_candidate(metrics, ["A", "B", "F"], flags) == "A"
    metrics_tie = {"B": _metric([0.5, 0.5]), "F": _metric([0.5, 0.5])}
    assert driver.select_candidate(metrics_tie, ["F", "B"], flags) == "B"  # no embedding first
    assert driver.select_candidate(metrics, ["A", "B"], flags, indices=[1]) == "B"


def test_leave_one_repo_out_selection_chooses_without_the_held_out_repo() -> None:
    strata = ["r1", "r1", "r2", "r2"]
    metrics = {"A": _metric([1.0, 1.0, 0.0, 0.0]), "B": _metric([0.0, 0.0, 0.5, 0.5])}
    flags = {"A": False, "B": False}
    result = driver.leave_one_repo_out_selection(strata, metrics, ["A", "B"], flags)
    # Without r1, B is best (0.5 vs 0.0) and is used for r1 (0.0); without r2, A is used for r2.
    assert result == [0.0, 0.0, 0.0, 0.0]


def _rules(**overrides: object) -> dict[str, object]:
    strata = ["r1"] * 10 + ["r2"] * 10 + ["r3"] * 10
    v5 = [0.4] * 30
    arguments: dict[str, object] = {
        "best_mrr": [0.6] * 30,
        "v5_mrr": v5,
        "strata": strata,
        "loro_mrr": [0.6] * 30,
        "cv_mrr": 0.55,
        "v5_train_mrr": 0.5,
        "p95_seconds": 1.0,
        "peak_rss_gib": 1.0,
    }
    arguments.update(overrides)
    return driver.apply_rules(**arguments)  # type: ignore[arg-type]


def test_rules_all_pass_when_every_condition_holds() -> None:
    result = _rules()
    assert result["all_pass"] is True
    assert result["R1_gain_over_v5"]["pass"] is True  # type: ignore[index]


@pytest.mark.parametrize(
    ("override", "rule"),
    [
        ({"best_mrr": [0.41] * 30, "loro_mrr": [0.6] * 30}, "R1_gain_over_v5"),
        ({"loro_mrr": [0.41] * 30}, "R3_selection_bias_check"),
        ({"cv_mrr": 0.5}, "R4_train_cross_validation"),
        ({"p95_seconds": 6.0}, "R5_speed_and_memory"),
        ({"peak_rss_gib": 9.0}, "R5_speed_and_memory"),
        ({"best_mrr": [0.9] * 10 + [0.2] * 10 + [0.9] * 10}, "R2_no_repo_regression"),
    ],
)
def test_each_rule_can_fail_on_its_own(override: dict[str, object], rule: str) -> None:
    result = _rules(**override)
    assert result[rule]["pass"] is False  # type: ignore[index]
    assert result["all_pass"] is False


def test_split_paths_keep_train_and_dev_apart() -> None:
    root = Path("/corpus")
    assert driver.split_paths(root, "dev")["trees"] == root / "dev" / "tree_manifest.json"
    assert driver.split_paths(root, "train")["trees"] == root / "train-eval" / "tree_manifest.json"
    with pytest.raises(ValueError):
        driver.split_paths(root, "holdout")
