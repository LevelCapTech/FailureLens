"""Public synthetic checks for the explicit cloud publication boundary."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import hf_jobs_release as release


def payload() -> dict:
    result = {"dataset_sha256": release.DATASET_SHA256}
    for split, repo in [("train", "pallets/click"), ("val", "python-attrs/attrs")]:
        result[split] = [
            {
                "record_id": f"synthetic-{split}-{i}",
                "query": "ValueError: empty input",
                "positive": "def positive(): return None",
                "negative": "def negative(): return True",
                "source_repository": repo,
                "dataset_version": "public-synthetic-test",
                "split": split,
            }
            for i in range(6)
        ]
    return result


@pytest.mark.parametrize(
    "feature", ["", "/home/private/source.py", "# issue17 marker", "injected fault"]
)
def test_unsanitized_features_rejected(feature: str) -> None:
    data = payload()
    data["train"][0]["query"] = feature
    with pytest.raises(ValueError):
        release.validate_payload(data)


def test_training_repository_separation_is_enforced() -> None:
    data = payload()
    data["train"][0]["source_repository"] = "psf/requests"
    with pytest.raises(ValueError, match="repository split"):
        release.validate_payload(data)


def test_model_code_must_match_the_audited_bytes() -> None:
    with pytest.raises(ValueError, match="code digest"):
        release.patch_modeling(release.MASK_ORIGINAL)


def test_cloud_bootstrap_uses_a_virtual_environment() -> None:
    check = (
        "import sys; assert sys.prefix != sys.base_prefix; "
        "import pip; print('isolated environment ready')"
    )
    result = subprocess.run(
        [sys.executable, "-c", release.CLOUD_BOOTSTRAP, check],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "isolated environment ready" in result.stdout


@pytest.mark.parametrize("seconds", [0, -1, 1801])
def test_retry_cannot_increase_the_approved_budget(tmp_path: Path, seconds: int) -> None:
    with pytest.raises(ValueError, match="timeout"):
        release.submit(tmp_path / "unused.gz", tmp_path / "unused.json", timeout_seconds=seconds)


def test_submit_keeps_features_and_token_in_secrets_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    private_token = "synthetic-test-token-never-a-real-credential"
    compressed = gzip.compress(json.dumps(payload()).encode())
    payload_path = tmp_path / "private.json.gz"
    payload_path.write_bytes(compressed)
    state_path = tmp_path / "job.json"
    calls = []

    class FakeApi:
        def whoami(self) -> dict:
            return {"name": "LevelCapTech"}

        def _validate_yaml(self, content: str, *, repo_type: str) -> None:
            assert repo_type == "model"
            assert content == release.MODEL_CARD_METADATA

        def model_info(self, repo: str) -> SimpleNamespace:
            assert repo == release.MODEL_REPO
            return SimpleNamespace(sha="a" * 40)

        def run_job(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(id="synthetic-job", url="https://example.invalid/job")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, get_token=lambda: private_token),
    )
    release.submit(payload_path, state_path)
    assert len(calls) == 1
    job = calls[0]
    public_spec = json.dumps({k: v for k, v in job.items() if k != "secrets"})
    assert private_token not in public_spec
    assert "synthetic-train-0" not in public_spec
    assert job["secrets"]["HF_TOKEN"] == private_token
    assert job["attempts"] == 1
    assert job["timeout"] == "30m"
    assert job["flavor"] == "t4-small"
    assert private_token not in state_path.read_text()
    assert private_token not in capsys.readouterr().out
    # A continuation must inspect the accepted Job instead of starting it again.
    with pytest.raises(FileExistsError, match="existing Job"):
        release.submit(payload_path, state_path)
    assert len(calls) == 1


def test_invalid_card_is_rejected_before_starting_a_paid_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload_path = tmp_path / "private.json.gz"
    payload_path.write_bytes(gzip.compress(json.dumps(payload()).encode()))

    class RejectingApi:
        def whoami(self) -> dict:
            return {"name": "LevelCapTech"}

        def _validate_yaml(self, content: str, *, repo_type: str) -> None:
            raise ValueError("Rejected card metadata")

        def run_job(self, **kwargs: object) -> None:
            pytest.fail("A paid Job must not start after card validation fails")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=RejectingApi, get_token=lambda: "synthetic-test-token"),
    )
    with pytest.raises(ValueError, match="card metadata"):
        release.submit(payload_path, tmp_path / "job.json")


def test_model_card_uses_an_immutable_revision_and_discloses_pending_evaluation() -> None:
    revision = "b" * 40
    card = release.model_card(
        {
            "effective_max_seq_length": 8192,
            "training": {"run_metadata": {"device_name": "synthetic GPU"}},
            "job_id": "synthetic-job",
            "artifact_sha256": "c" * 64,
        },
        revision,
    )
    assert f'revision = "{revision}"' in card
    assert "Independent held-out evaluation (Gate C) is pending" in card
    assert "No improvement over" in card
    assert "PLACEHOLDER" not in card
