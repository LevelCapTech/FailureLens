"""Synthetic-only complete-tree and shipped-retrieval parity tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from evaluation.independent_runner import MethodSpec, run_independent
from evaluation.private_adapter import PrivateInputError, RankResult, compare_rankers
from evaluation.product_pool import load_product_cases


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, frames: int = 2, function: bool = True, prefixed: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base"
    base.mkdir()
    clean = "def cause():\n    return 1\n\ndef decoy():\n    return 2\n"
    failed = clean.replace("return 1", "return 9")
    relative = "src/mod.py" if prefixed else "mod.py"
    (base / relative).parent.mkdir(parents=True, exist_ok=True)
    (base / relative).write_text(clean)
    subprocess.run(["git", "init", "-q", str(base)], check=True)
    subprocess.run(["git", "-C", str(base), "add", relative], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(base),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(base), "rev-parse", "HEAD"], text=True
    ).strip()
    root = tmp_path / "failing"
    root.mkdir()
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).write_text(failed)
    log = "Traceback (most recent call last):\n"
    if frames:
        log += '  File "mod.py", line 2, in cause\n'
    if frames == 2:
        log += '  File "mod.py", line 5, in decoy\n'
    log += "AssertionError: synthetic failure"
    first = {"file": "mod.py", "start_line": 1, "end_line": 2}
    second = {"file": "mod.py", "start_line": 4, "end_line": 5}
    if function:
        first["function"] = "cause"
        second["function"] = "decoy"
    record = {
        "record_id": "synthetic-case",
        "dataset_id": "synthetic-eval",
        "source_repository": "synthetic/repo",
        "source_commit": commit,
        "source_license": "MIT",
        "language": "python",
        "test_command": "python -m unittest",
        "pre_change_result": {"status": "passed"},
        "post_change_result": {"status": "failed"},
        "error_log": log,
        "stack_trace": log if frames else None,
        "candidates": [
            {"location": first, "label": "positive"},
            {"location": second, "label": "negative"},
        ],
        "label_rationale": "Synthetic behavior changed inside cause, not the surfacing site.",
        "generated_at": "2026-01-01T00:00:00Z",
        "generator_version": "synthetic-v1",
    }
    records = tmp_path / "records.json"
    source = tmp_path / "source.json"
    train = tmp_path / "train.json"
    trees = tmp_path / "trees.json"
    _write(records, [record])
    _write(
        source,
        {
            "repositories": {"synthetic/repo": {"commit": commit, "rights_cleared": True}},
            "snapshots": {
                "synthetic-case": {
                    "mod.py": {
                        "path": str(root / relative),
                        "sha256": hashlib.sha256(failed.encode()).hexdigest(),
                    }
                }
            },
        },
    )
    _write(train, {"approval_id": "synthetic-only", "training_repositories": ["synthetic/train"]})
    _write(
        trees,
        {
            "cases": {
                "synthetic-case": {
                    "base_repository": str(base),
                    "root": str(root),
                    "commit": commit,
                    "indexed_files": {relative: hashlib.sha256(failed.encode()).hexdigest()},
                    "mutation": {
                        "file": relative,
                        "function": "cause",
                        "old": "return 1",
                        "new": "return 9",
                    },
                }
            }
        },
    )
    return records, source, train, trees, root


def _load(paths):
    records, source, train, trees, _ = paths
    return load_product_cases(
        records,
        hashlib.sha256(records.read_bytes()).hexdigest(),
        source,
        train,
        trees,
        top_k=2,
        production=False,
    )


def test_product_pool_and_exact_unlabeled_parity(tmp_path: Path) -> None:
    cases, baseline = _load(_fixture(tmp_path))
    (case,) = cases
    seen = []

    def model(request):
        seen.append(request)
        assert not hasattr(request, "positive_ids")
        assert all(not hasattr(candidate, "label") for candidate in request.candidates)
        return RankResult(
            tuple(reversed([c.id for c in request.candidates])), request.digest, request.top_k
        )

    first, second = compare_rankers(case, baseline, model)
    assert seen == [case.request]
    assert len(case.request.candidates) == 2
    assert first.ordered_ids == tuple(c.id for c in case.request.candidates)
    assert second.ordered_ids == tuple(reversed(first.ordered_ids))
    assert all(
        c.source_sha256 == hashlib.sha256(c.source_text.encode()).hexdigest()
        for c in case.request.candidates
    )
    assert case.positive_function_ids == frozenset({"mod.py::cause"})
    assert json.loads(case.request.query)["error_log"].startswith("Traceback")


@pytest.mark.parametrize("change", ["digest", "missing", "extra", "mutation", "changed_commit"])
def test_full_tree_fail_closed(tmp_path: Path, change: str) -> None:
    paths = _fixture(tmp_path)
    records, _, _, trees, root = paths
    manifest = json.loads(trees.read_text())
    item = manifest["cases"]["synthetic-case"]
    if change == "digest":
        item["indexed_files"]["mod.py"] = "0" * 64
    elif change == "missing":
        item["indexed_files"] = {}
    elif change == "extra":
        (root / "extra.py").write_text("pass\n")
    elif change == "mutation":
        item["mutation"]["old"] = "not present"
    else:
        item["commit"] = "0" * 40
    _write(trees, manifest)
    with pytest.raises(PrivateInputError) as error:
        _load(paths)
    assert "synthetic-case" not in str(error.value)
    assert str(records) not in str(error.value)
    assert "return 9" not in str(error.value)


def test_absent_positive_scores_zero_and_missing_function_na(tmp_path: Path) -> None:
    from evaluation.independent_runner import _scores

    cases, baseline = _load(_fixture(tmp_path, frames=1))
    (case,) = cases
    assert _scores(case, baseline(case.request))["top1"] == 1.0
    # The single retrieved function is now the decoy, so the positive is outside the pool.
    paths = _fixture(tmp_path / "absent", frames=2)
    records = json.loads(paths[0].read_text())
    records[0]["error_log"] = (
        'Traceback (most recent call last):\n  File "mod.py", line 5, in decoy\nAssertionError: synthetic failure'
    )
    _write(paths[0], records)
    cases, baseline = _load(paths)
    (case,) = cases
    assert _scores(case, baseline(case.request))["top1"] == 0.0
    assert _scores(case, baseline(case.request))["mrr"] == 0.0
    paths = _fixture(tmp_path / "no-function", frames=1, function=False)
    cases, baseline = _load(paths)
    assert _scores(cases[0], baseline(cases[0].request))["top1"] is None


def test_empty_product_retrieval_keeps_denominator_for_both(tmp_path: Path) -> None:
    cases, baseline = _load(_fixture(tmp_path, frames=0))
    assert cases[0].request.candidates == ()

    def fake_factory(seed):
        def model(request):
            raise AssertionError("empty pool must not invoke model")

        return model

    metadata = {
        "code_sha": "a" * 40,
        "data_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "train_manifest_sha256": "d" * 64,
        "model_sha256": "e" * 64,
        "python_version": "3.11",
        "hardware": "synthetic",
        "command": "run_independent-api",
    }
    report = run_independent(
        cases,
        fake_factory,
        baseline=baseline,
        baseline_spec=MethodSpec("baseline-stacktrace-lexical", "6c9c1c0", 0, 0),
        model_spec=MethodSpec("synthetic-model", "v0", 1, 1),
        metadata=metadata,
    )
    assert report["counts"]["cases"] == 1
    assert report["counts"]["retrieval_abstentions"] == 1
    assert report["counts"]["retrieval_abstention_rate"] == 1.0
    assert report["counts"]["positive_function_absent_from_pool"] == 1
    assert report["counts"]["retrieved_pool_function_recall"] == 0.0
    for run in report["runs"]:
        assert run["metrics"]["baseline"]["top1"] == 0.0
        assert run["metrics"]["model"]["top1"] == 0.0


def test_label_file_alias_maps_uniquely_to_full_tree(tmp_path: Path) -> None:
    cases, baseline = _load(_fixture(tmp_path, prefixed=True))
    (case,) = cases
    assert case.positive_file_ids == frozenset({"src/mod.py"})
    assert case.positive_function_ids == frozenset({"src/mod.py::cause"})
    assert all(candidate.file_id == "src/mod.py" for candidate in case.request.candidates)
    assert baseline(case.request).ordered_ids == tuple(c.id for c in case.request.candidates)


@pytest.mark.parametrize("function", ["missing", "duplicate"])
def test_product_function_mapping_fails_closed(tmp_path: Path, function: str) -> None:
    from types import SimpleNamespace

    from evaluation.product_pool import _product_candidate

    from failurelens import FailureLens

    root = tmp_path / "source"
    root.mkdir()
    source = "def cause():\n    return 1\n"
    if function == "duplicate":
        source += "\ndef cause():\n    return 2\n"
    (root / "mod.py").write_text(source)
    index = FailureLens.from_baseline().index(root)
    output = SimpleNamespace(
        file="mod.py",
        function="cause" if function == "duplicate" else "absent",
        start_line=1,
        end_line=2,
    )
    with pytest.raises(PrivateInputError, match="function span"):
        _product_candidate(output, root, index)


def test_missing_non_candidate_indexed_file_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _, _, _, trees, root = paths
    manifest = json.loads(trees.read_text())
    item = manifest["cases"]["synthetic-case"]
    base = Path(item["base_repository"])
    (base / "other.py").write_text("def unrelated():\n    pass\n")
    subprocess.run(["git", "-C", str(base), "add", "other.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(base),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "add unrelated file",
        ],
        check=True,
    )
    updated = subprocess.check_output(
        ["git", "-C", str(base), "rev-parse", "HEAD"], text=True
    ).strip()
    records = json.loads(paths[0].read_text())
    records[0]["source_commit"] = updated
    _write(paths[0], records)
    source = json.loads(paths[1].read_text())
    source["repositories"]["synthetic/repo"]["commit"] = updated
    _write(paths[1], source)
    item["commit"] = updated
    item["indexed_files"]["other.py"] = hashlib.sha256(b"def unrelated():\n    pass\n").hexdigest()
    _write(trees, manifest)
    assert not (root / "other.py").exists()
    with pytest.raises(PrivateInputError, match="partial failing tree"):
        _load(paths)


def test_committed_non_utf8_indexed_file_is_attested_product_skip(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _, _, _, trees, root = paths
    manifest = json.loads(trees.read_text())
    item = manifest["cases"]["synthetic-case"]
    base = Path(item["base_repository"])
    (base / "legacy.py").write_bytes(b"# nonutf8 \xff\n")
    (root / "legacy.py").write_bytes(b"# nonutf8 \xff\n")
    subprocess.run(["git", "-C", str(base), "add", "legacy.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(base),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-qm",
            "add legacy file",
        ],
        check=True,
    )
    updated = subprocess.check_output(
        ["git", "-C", str(base), "rev-parse", "HEAD"], text=True
    ).strip()
    records = json.loads(paths[0].read_text())
    records[0]["source_commit"] = updated
    _write(paths[0], records)
    source = json.loads(paths[1].read_text())
    source["repositories"]["synthetic/repo"]["commit"] = updated
    _write(paths[1], source)
    item["commit"] = updated
    item["indexed_files"]["legacy.py"] = hashlib.sha256(b"# nonutf8 \xff\n").hexdigest()
    _write(trees, manifest)
    cases, baseline = _load(paths)
    assert len(cases[0].request.candidates) == 2
    assert baseline(cases[0].request).ordered_ids
