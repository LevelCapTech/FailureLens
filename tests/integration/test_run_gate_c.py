"""Synthetic-only tests for the aggregate-only Gate C driver."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
from evaluation import run_gate_c
from evaluation.private_adapter import PrivateInputError, RankResult
from test_product_pool import _fixture


def _artifact(tmp_path: Path, *, train: str = "a" * 64, params: int = 1000) -> tuple[Path, str]:
    folder = tmp_path / "model"
    folder.mkdir()
    weights = folder / "model_final.safetensors"
    weights.write_bytes(b"synthetic-weights")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    meta = {
        "artifact_sha256": digest,
        "run_metadata": {
            "model_id": run_gate_c.MODEL_ID,
            "model_revision": run_gate_c.MODEL_REVISION,
            "dataset_sha256": train,
            "total_params": params,
        },
    }
    (folder / "experiment_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return weights, digest


def _args(tmp_path: Path, weights: Path, digest: str, paths) -> list[str]:
    records, source, train, trees, _ = paths
    return [
        "--records", str(records),
        "--records-sha256", hashlib.sha256(records.read_bytes()).hexdigest(),
        "--source-manifest", str(source),
        "--train-manifest", str(train),
        "--tree-manifest", str(trees),
        "--model-artifact", str(weights),
        "--model-sha256", digest,
        "--model-label", "synthetic-v1",
        "--out-dir", str(tmp_path / "out"),
        "--top-k", "2",
        "--dev",
    ]  # fmt: skip


def _fake_factory(*_: object):
    def factory(seed: int):
        def model(request):
            return RankResult(
                tuple(sorted(c.id for c in request.candidates)), request.digest, request.top_k
            )

        return model

    return factory


def test_verify_candidate_accepts_matching_provenance(tmp_path: Path) -> None:
    weights, digest = _artifact(tmp_path)
    spec, dataset = run_gate_c.verify_candidate(weights, digest, "a" * 64)
    assert (spec.revision, spec.parameter_count, dataset) == (digest, 1000, "a" * 64)


@pytest.mark.parametrize("change", ["digest", "train", "params", "meta"])
def test_verify_candidate_fails_closed(tmp_path: Path, change: str) -> None:
    weights, digest = _artifact(tmp_path, params=2_000_000_000 if change == "params" else 1000)
    expected_train = "b" * 64 if change == "train" else "a" * 64
    if change == "digest":
        digest = "0" * 64
    if change == "meta":
        (weights.parent / "experiment_metadata.json").unlink()
    with pytest.raises(PrivateInputError) as error:
        run_gate_c.verify_candidate(weights, digest, expected_train)
    assert str(tmp_path) not in str(error.value)


def test_end_to_end_is_aggregate_only_and_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path / "fx")
    weights, digest = _artifact(tmp_path)
    monkeypatch.setattr(run_gate_c, "make_candidate_factory", _fake_factory)
    monkeypatch.setattr(run_gate_c, "_git_head", lambda: ("c" * 40, False))
    seen: list[tuple[dict, str]] = []

    code = run_gate_c.main(
        _args(tmp_path, weights, digest, paths), hook=lambda r, label: seen.append((r, label))
    )

    assert code == 0
    (report_path,) = (tmp_path / "out").glob("dev-product-synthetic-v1-*.json")
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    report = json.loads(report_path.read_text())
    assert report["mode"] == "dev" and report["gate_c_decision"] is None
    assert report["counts"]["cases"] == 1
    assert report["methods"]["baseline"]["name"] == "shipped-baseline-stacktrace-lexical"
    assert [label for _, label in seen] == ["synthetic-v1"]
    text = report_path.read_text() + capsys.readouterr().out
    for private in ("synthetic-case", "return 9", "mod.py", "Traceback", str(tmp_path)):
        assert private not in text.replace(report_path.name, "")


def test_dirty_tree_is_rejected_without_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _fixture(tmp_path / "fx")
    weights, digest = _artifact(tmp_path)
    monkeypatch.setattr(run_gate_c, "_git_head", lambda: ("c" * 40, True))
    assert run_gate_c.main(_args(tmp_path, weights, digest, paths)) == 4
    assert "uncommitted" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_production_floor_applies_without_dev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "fx")
    weights, digest = _artifact(tmp_path)
    monkeypatch.setattr(run_gate_c, "_git_head", lambda: ("c" * 40, False))
    argv = [a for a in _args(tmp_path, weights, digest, paths) if a != "--dev"]
    assert run_gate_c.main(argv) == 4


def test_record_hook_is_noop_by_default_and_forwards_when_set() -> None:
    report = {"x": 1}
    assert run_gate_c.record_evaluation(report, "v", None) is False
    calls: list[str] = []
    assert run_gate_c.record_evaluation(report, "v", lambda r, label: calls.append(label)) is True
    assert calls == ["v"]


@pytest.mark.parametrize("label", ["", "has space", "/etc/passwd", "private-run", "x" * 61])
def test_unsafe_labels_are_rejected(label: str) -> None:
    with pytest.raises(PrivateInputError):
        run_gate_c._label(label)


def test_hardware_token_is_allowlist_safe() -> None:
    token = run_gate_c.hardware_token("cpu")
    assert token and len(token) <= 80
    assert all(c.isalnum() or c in "_.+-" for c in token)


def test_supplied_pool_mode_uses_labeled_pools_and_names_comparator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path / "fx")
    weights, digest = _artifact(tmp_path)
    monkeypatch.setattr(run_gate_c, "make_candidate_factory", _fake_factory)
    monkeypatch.setattr(run_gate_c, "_git_head", lambda: ("c" * 40, False))
    argv = [*_args(tmp_path, weights, digest, paths), "--pool", "supplied"]
    assert run_gate_c.main(argv) == 0
    (report_path,) = (tmp_path / "out").glob("dev-supplied-*.json")
    report = json.loads(report_path.read_text())
    assert report["pool"] == "supplied"
    assert report["methods"]["baseline"]["name"] == "supplied-pool-stack-lexical"
    assert report["counts"]["retrieval_abstentions"] == 0
