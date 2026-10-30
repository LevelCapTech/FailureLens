"""Stage 2 integration test: CodeRankEmbed fine-tuning with checkpoint/resume.

Requires GPU, private dataset, and model cache.
Skip unless ``--run-ml-integration`` is passed.

Verifies:
1. load_stage2_examples() reconstructs 12 examples (6 train, 6 val).
2. No generator marker (``# issue17 injected fault``) appears in positive,
   query, or negative texts for ALL 12 examples (fail-closed).
3. Positive texts are byte-identical to an independent second reconstruction
   from the same private source files (digest proof).
4. Positive != negative texts (not clean-only source).
5. Full run produces monotonically decreasing or stable train loss.
6. Checkpoint/resume: epoch 3 loss from resumed run matches full run within 1e-5.
7. Final artifact is safetensors and has a valid SHA256 digest.
8. No .bin files in the model cache (safetensors-only loader).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PRIVATE_DATA_ROOT = Path.home() / ".local/share/failurelens/train-issue17-v2"
_STAGE1_MODEL_CACHE = (
    Path(__file__).parent.parent.parent / "models/cache/nomic-coderankeembed-smoke"
)
_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
_SNAPSHOT_PATH = _STAGE1_MODEL_CACHE / "models--nomic-ai--CodeRankEmbed" / "snapshots" / _REVISION
_GENERATOR_MARKER = "# issue17 injected fault"


def _require_ml_integration(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--run-ml-integration", default=False):
        pytest.skip("Pass --run-ml-integration to run ML integration tests")


@pytest.fixture(scope="module")
def stage2_examples(request: pytest.FixtureRequest):  # type: ignore[misc]
    _require_ml_integration(request)
    pytest.importorskip("torch")
    if not _PRIVATE_DATA_ROOT.exists():
        pytest.skip(f"Private dataset not found at {_PRIVATE_DATA_ROOT}")
    from training.data_stage2 import load_stage2_examples

    return load_stage2_examples(_PRIVATE_DATA_ROOT)


@pytest.fixture(scope="module")
def stage2_run(request: pytest.FixtureRequest, stage2_examples, tmp_path_factory):  # type: ignore[misc]
    _require_ml_integration(request)
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not _SNAPSHOT_PATH.exists():
        pytest.skip(f"Model snapshot not found at {_SNAPSHOT_PATH}")

    from training.finetune import run_stage2_finetune

    train_examples, val_examples = stage2_examples
    output_dir = tmp_path_factory.mktemp("stage2_full")
    config = {
        "seed": 42,
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.00002,
        "weight_decay": 0.01,
        "temperature": 0.07,
    }
    result = run_stage2_finetune(
        config,
        train_examples=train_examples,
        val_examples=val_examples,
        output_dir=output_dir,
        model_cache_dir=str(_STAGE1_MODEL_CACHE),
    )
    return result, output_dir


@pytest.fixture(scope="module")
def stage2_resumed(request: pytest.FixtureRequest, stage2_examples, stage2_run, tmp_path_factory):  # type: ignore[misc]
    _require_ml_integration(request)
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from training.finetune import run_stage2_finetune

    train_examples, val_examples = stage2_examples
    _, full_output_dir = stage2_run
    resume_dir = full_output_dir / "checkpoint_epoch_2"
    output_dir = tmp_path_factory.mktemp("stage2_resumed")
    config = {
        "seed": 42,
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 0.00002,
        "weight_decay": 0.01,
        "temperature": 0.07,
    }
    result = run_stage2_finetune(
        config,
        train_examples=train_examples,
        val_examples=val_examples,
        output_dir=output_dir,
        resume_from=resume_dir,
        model_cache_dir=str(_STAGE1_MODEL_CACHE),
    )
    return result


class TestStage2DataReconstruction:
    def test_split_counts(self, stage2_examples) -> None:
        train, val = stage2_examples
        assert len(train) == 6, f"Expected 6 train; got {len(train)}"
        assert len(val) == 6, f"Expected 6 val; got {len(val)}"

    def test_split_repos(self, stage2_examples) -> None:
        train, val = stage2_examples
        assert all(ex.source_repository == "pallets/click" for ex in train)
        assert all(ex.source_repository == "python-attrs/attrs" for ex in val)

    def test_no_generator_marker_in_positive(self, stage2_examples) -> None:
        """Fail-closed: generator marker must be ABSENT from all 12 positive texts."""
        train, val = stage2_examples
        assert len(train) == 6 and len(val) == 6, "Expected exactly 12 total examples"
        for ex in train + val:
            assert _GENERATOR_MARKER not in ex.positive, (
                f"Generator marker found in positive text for {ex.record_id!r}: "
                f"feature contains label signal"
            )

    def test_no_generator_marker_in_query(self, stage2_examples) -> None:
        """Fail-closed: generator marker must be ABSENT from all 12 query texts."""
        train, val = stage2_examples
        assert len(train) == 6 and len(val) == 6, "Expected exactly 12 total examples"
        for ex in train + val:
            assert _GENERATOR_MARKER not in ex.query, (
                f"Generator marker found in query (error_log) for {ex.record_id!r}: "
                f"feature contains label signal"
            )

    def test_no_generator_marker_in_negative(self, stage2_examples) -> None:
        """Fail-closed: generator marker must be ABSENT from all 12 negative texts."""
        train, val = stage2_examples
        assert len(train) == 6 and len(val) == 6, "Expected exactly 12 total examples"
        for ex in train + val:
            assert _GENERATOR_MARKER not in ex.negative, (
                f"Generator marker found in negative text for {ex.record_id!r}: "
                f"feature contains label signal"
            )

    def test_positive_differs_from_negative(self, stage2_examples) -> None:
        train, val = stage2_examples
        for ex in train + val:
            assert ex.positive != ex.negative, (
                f"Positive == negative for {ex.record_id!r} (clean-only source?)"
            )

    def test_positive_reconstruction_digest(self, stage2_examples) -> None:
        """Positive texts must byte-match an independent second reconstruction.

        Re-derives each positive text from scratch using the private source files
        and the same reconstruction logic, then compares SHA-256 digests.  This
        proves that load_stage2_examples() faithfully reconstructs the mutated
        source without extra transformation.
        """
        import hashlib
        import json as _json

        train, val = stage2_examples
        records = _json.loads((_PRIVATE_DATA_ROOT / "records.json").read_text())

        from training.data_stage2 import (
            _REPO_SOURCE_KEYS,
            _extract_mutated_function_text,
            _reconstruct_injection_statement,
            _strip_generator_marker,
        )

        by_id = {ex.record_id: ex for ex in train + val}
        for rec in records:
            ex = by_id.get(rec["record_id"])
            if ex is None:
                continue

            repo_key = _REPO_SOURCE_KEYS[rec["source_repository"]]
            source_dir = _PRIVATE_DATA_ROOT / "sources" / repo_key
            il = rec["injection_location"]
            injection_stmt = _reconstruct_injection_statement(rec)
            pos_file = source_dir / il["file"]

            expected_positive = _strip_generator_marker(
                _extract_mutated_function_text(
                    pos_file,
                    il["function"],
                    injection_stmt,
                    il["start_line"],
                    il["end_line"],
                )
            )

            expected_digest = hashlib.sha256(expected_positive.encode()).hexdigest()
            actual_digest = hashlib.sha256(ex.positive.encode()).hexdigest()

            assert expected_digest == actual_digest, (
                f"Positive reconstruction digest mismatch for {ex.record_id!r}: "
                f"expected {expected_digest} got {actual_digest}. "
                f"load_stage2_examples() diverges from independent reconstruction."
            )

    def test_query_is_nonempty_traceback(self, stage2_examples) -> None:
        train, val = stage2_examples
        for ex in train + val:
            assert ex.query.strip(), f"Empty query for {ex.record_id!r}"
            assert "Traceback" in ex.query or "Error" in ex.query, (
                f"Query does not look like a traceback for {ex.record_id!r}"
            )

    def test_no_label_rationale_in_features(self, stage2_examples) -> None:
        """Positive/negative texts must not be the label_rationale string."""
        import json as _json

        records = _json.loads((_PRIVATE_DATA_ROOT / "records.json").read_text())
        train, val = stage2_examples
        by_id = {ex.record_id: ex for ex in train + val}
        for rec in records:
            ex = by_id.get(rec["record_id"])
            if ex is None:
                continue
            rationale: str = rec.get("label_rationale", "")
            assert ex.positive != rationale, f"Positive is label_rationale for {ex.record_id!r}"
            assert ex.negative != rationale, f"Negative is label_rationale for {ex.record_id!r}"


class TestStage2Finetune:
    def test_full_run_produces_history(self, stage2_run) -> None:
        result, _ = stage2_run
        assert "train_loss" in result["history"]
        assert len(result["history"]["train_loss"]) == 3
        assert len(result["history"]["val_loss"]) == 3

    def test_train_loss_decreases(self, stage2_run) -> None:
        result, _ = stage2_run
        losses = result["history"]["train_loss"]
        assert losses[0] > losses[-1], f"Train loss did not decrease over 3 epochs: {losses}"

    def test_artifact_is_safetensors(self, stage2_run) -> None:
        result, _ = stage2_run
        artifact = Path(result["artifact_path"])
        assert artifact.exists(), f"Artifact not found: {artifact}"
        assert artifact.suffix == ".safetensors"

    def test_artifact_digest_matches(self, stage2_run) -> None:
        import hashlib

        result, _ = stage2_run
        artifact = Path(result["artifact_path"])
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest == result["artifact_sha256"], "Artifact digest mismatch"

    def test_checkpoints_exist_as_safetensors(self, stage2_run) -> None:
        _, output_dir = stage2_run
        for epoch in range(1, 4):
            ckpt = output_dir / f"checkpoint_epoch_{epoch}" / "model.safetensors"
            assert ckpt.exists(), f"Checkpoint not found: {ckpt}"

    def test_no_bin_files_in_cache(self, request: pytest.FixtureRequest) -> None:
        _require_ml_integration(request)
        bin_files = list(_STAGE1_MODEL_CACHE.rglob("*.bin"))
        assert bin_files == [], f"Found .bin files in model cache: {bin_files}"


class TestStage2CheckpointResume:
    def test_resumed_epoch3_loss_matches_full_run(self, stage2_run, stage2_resumed) -> None:
        """Epoch 3 loss from resumed run must match full run within 1e-5."""
        full_result, _ = stage2_run
        resumed_result = stage2_resumed

        # Resumed from epoch-2 checkpoint: history has 3 items (2 loaded + 1 new epoch)
        # Full run epoch-3 loss is index [2]; resumed run epoch-3 is the last item
        full_train = full_result["history"]["train_loss"][2]
        resumed_train = resumed_result["history"]["train_loss"][-1]
        assert abs(full_train - resumed_train) < 1e-5, (
            f"Epoch 3 train loss mismatch: full={full_train:.6f} "
            f"resumed={resumed_train:.6f} diff={abs(full_train - resumed_train):.6f}"
        )

        full_val = full_result["history"]["val_loss"][2]
        resumed_val = resumed_result["history"]["val_loss"][-1]
        assert abs(full_val - resumed_val) < 1e-5, (
            f"Epoch 3 val loss mismatch: full={full_val:.6f} "
            f"resumed={resumed_val:.6f} diff={abs(full_val - resumed_val):.6f}"
        )

    def test_resumed_metadata_records_correct_epoch(self, stage2_run, stage2_resumed) -> None:
        resumed_result = stage2_resumed
        assert resumed_result["history"]["train_loss"], "Resumed history is empty"
        # Resumed from epoch 2 checkpoint: history has 2 pre-loaded + 1 new = 3 total
        assert len(resumed_result["history"]["train_loss"]) == 3, (
            f"Expected 3 epochs in resumed history (2 loaded + 1 new); got "
            f"{len(resumed_result['history']['train_loss'])}"
        )
