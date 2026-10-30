"""Public synthetic-only contract tests for the private evaluation boundary."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from evaluation.private_adapter import (
    PrivateInputError,
    RankResult,
    compare_rankers,
    load_private_cases,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, *, count: int = 1, repos: int = 1, function: bool = True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    records = []
    sources = {}
    for index in range(repos):
        root = tmp_path / f"source{index}"
        root.mkdir()
        (root / "mod.py").write_text("def cause():\n    return 1\n\ndef decoy():\n    return 2\n")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "mod.py"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
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
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        sources[f"synthetic/repo{index}"] = {
            "path": str(root),
            "commit": commit,
            "rights_cleared": True,
        }
    for index in range(count):
        repo_index = index % repos
        repo = f"synthetic/repo{repo_index}"
        first = {"file": "mod.py", "start_line": 1, "end_line": 2}
        second = {"file": "mod.py", "start_line": 4, "end_line": 5}
        if function:
            first["function"] = "cause"
            second["function"] = "decoy"
        records.append(
            {
                "record_id": f"private-case-{index}",
                "dataset_id": "synthetic-eval",
                "source_repository": repo,
                "source_commit": sources[repo]["commit"],
                "source_license": "MIT",
                "language": "python",
                "test_command": "python -m unittest",
                "pre_change_result": {"status": "passed"},
                "post_change_result": {"status": "failed"},
                "error_log": f"AssertionError: synthetic failure {index}",
                "candidates": [
                    {"location": first, "label": "positive"},
                    {"location": second, "label": "negative"},
                ],
                "label_rationale": "Synthetic cause changes behavior, not the surfacing site.",
                "generated_at": "2026-01-01T00:00:00Z",
                "generator_version": "synthetic-v1",
            }
        )
    records_path = tmp_path / "private-records.json"
    sources_path = tmp_path / "sources.json"
    train_path = tmp_path / "train.json"
    _write(records_path, records)
    _write(sources_path, {"repositories": sources})
    _write(
        train_path, {"approval_id": "fixture-only", "training_repositories": ["synthetic/train"]}
    )
    return records, records_path, sources_path, train_path


def _load(paths: tuple[Path, Path, Path], *, production: bool = False):
    records_path, sources_path, train_path = paths
    digest = hashlib.sha256(records_path.read_bytes()).hexdigest()
    return load_private_cases(
        records_path, digest, sources_path, train_path, top_k=2, production=production
    )


def test_parity_and_no_label_to_rankers(tmp_path: Path) -> None:
    _, *paths = _fixture(tmp_path)
    (case,) = _load(tuple(paths))
    seen = []

    def ranker(request):
        assert not hasattr(request, "positive_ids")
        assert not hasattr(request.candidates[0], "label")
        seen.append(request)
        return RankResult(tuple(c.id for c in request.candidates), request.digest, request.top_k)

    def reordered(request):
        seen.append(request)
        return RankResult(
            tuple(c.id for c in reversed(request.candidates)), request.digest, request.top_k
        )

    baseline, model = compare_rankers(case, ranker, reordered)
    assert seen[0] is seen[1] is case.request
    assert baseline.ordered_ids != model.ordered_ids
    assert case.positive_function_ids == frozenset({"mod.py::cause"})


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "digest", "top_k", "invalid"])
def test_parity_rejects_changed_results(tmp_path: Path, change: str) -> None:
    _, *paths = _fixture(tmp_path)
    (case,) = _load(tuple(paths))

    def valid(request):
        return RankResult(tuple(c.id for c in request.candidates), request.digest, request.top_k)

    def bad(request):
        result = valid(request)
        ids = result.ordered_ids
        if change == "missing":
            return replace(result, ordered_ids=ids[:1])
        if change == "extra":
            return replace(result, ordered_ids=(*ids, "extra"))
        if change == "duplicate":
            return replace(result, ordered_ids=(ids[0], ids[0]))
        if change == "digest":
            return replace(result, request_digest="0" * 64)
        if change == "top_k":
            return replace(result, top_k=1)
        return "not a ranking"

    with pytest.raises(PrivateInputError):
        compare_rankers(case, valid, bad)


def test_missing_function_is_na(tmp_path: Path) -> None:
    _, *paths = _fixture(tmp_path, function=False)
    (case,) = _load(tuple(paths))
    assert case.positive_function_ids is None
    assert case.positive_file_ids == frozenset({"mod.py"})


def test_production_minimums(tmp_path: Path) -> None:
    _, *paths = _fixture(tmp_path, count=29, repos=3)
    with pytest.raises(PrivateInputError, match="insufficient evaluation set"):
        _load(tuple(paths), production=True)
    _, *paths = _fixture(tmp_path / "other", count=30, repos=2)
    with pytest.raises(PrivateInputError, match="insufficient evaluation set"):
        _load(tuple(paths), production=True)


def test_production_accepts_synthetic_minimum(tmp_path: Path) -> None:
    _, *paths = _fixture(tmp_path, count=30, repos=3)
    assert len(_load(tuple(paths), production=True)) == 30


@pytest.mark.parametrize(
    "change", ["digest", "schema", "quality", "overlap", "id", "signature", "pin", "location"]
)
def test_fail_closed_without_private_values(tmp_path: Path, change: str) -> None:
    records, records_path, sources_path, train_path = _fixture(tmp_path, count=2)
    sources = json.loads(sources_path.read_text())
    train = json.loads(train_path.read_text())
    expected = "0" * 64 if change == "digest" else None
    if change == "schema":
        del records[0]["source_commit"]
    if change == "quality":
        records[0]["source_license"] = "unknown"
    if change == "overlap":
        train["training_repositories"] = ["synthetic/repo0"]
    if change == "id":
        records[1]["record_id"] = records[0]["record_id"]
    if change == "signature":
        records[1]["error_log"] = records[0]["error_log"]
    if change == "pin":
        sources["repositories"]["synthetic/repo0"]["commit"] = "0" * 40
    if change == "location":
        records[0]["candidates"][0]["location"]["start_line"] = 100
        records[0]["candidates"][0]["location"]["end_line"] = 101
    _write(records_path, records)
    _write(sources_path, sources)
    _write(train_path, train)
    digest = expected or hashlib.sha256(records_path.read_bytes()).hexdigest()
    with pytest.raises(PrivateInputError) as error:
        load_private_cases(
            records_path, digest, sources_path, train_path, top_k=2, production=False
        )
    message = str(error.value)
    assert "private-case" not in message
    assert str(records_path) not in message
    assert "synthetic/repo" not in message
    assert "positive" not in message


def test_changed_request_or_content_is_rejected(tmp_path: Path) -> None:
    _, *paths = _fixture(tmp_path)
    (case,) = _load(tuple(paths))

    def valid(request):
        return RankResult(tuple(c.id for c in request.candidates), request.digest, request.top_k)

    changed = replace(case.request.candidates[0], source_text="altered")
    tampered = replace(case.request, candidates=(changed, *case.request.candidates[1:]))
    with pytest.raises(PrivateInputError):
        compare_rankers(replace(case, request=tampered), valid, valid)

    def mutate(request):
        object.__setattr__(request, "top_k", 1)
        return valid(request)

    with pytest.raises(PrivateInputError):
        compare_rankers(case, mutate, valid)


def test_ranker_exception_is_redacted(tmp_path: Path) -> None:
    _, *paths = _fixture(tmp_path)
    (case,) = _load(tuple(paths))

    def leak(_request):
        raise RuntimeError("private-case-0 positive secret")

    with pytest.raises(PrivateInputError) as error:
        compare_rankers(case, leak, leak)
    assert "private-case" not in str(error.value)
    assert "positive" not in str(error.value)
