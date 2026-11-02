"""Offline regression tests for the read-only inspection boundary."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "inspect_release", Path(__file__).with_name("inspect_release.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SECRET = "synthetic-test-secret-not-a-real-credential"


class InspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls = []
        self.factory_options = []
        test = self

        class ReadOnlyApi:
            def inspect_job(self, *, job_id: str, namespace: str) -> SimpleNamespace:
                test.calls.append(("inspect", job_id, namespace))
                return SimpleNamespace(
                    id=job_id,
                    status=SimpleNamespace(
                        stage={"failed": "ERROR", "done": "COMPLETED", "live": "RUNNING"}[job_id]
                    ),
                    flavor="t4-small",
                    command=[SECRET],
                    secrets={"HF_TOKEN": SECRET},
                    environment={"PRIVATE": SECRET},
                )

            def model_info(self, repo: str, **kwargs: object) -> SimpleNamespace:
                test.calls.append(("model_info", repo, kwargs))
                return SimpleNamespace(
                    id=repo,
                    sha="a" * 40,
                    private=False,
                    gated=False,
                    siblings=[
                        SimpleNamespace(
                            rfilename="model.safetensors",
                            size=500_000_000,
                            lfs={"sha256": "b" * 64},
                        ),
                        SimpleNamespace(rfilename="config.json", size=25, lfs=None),
                    ],
                )

            def list_jobs_hardware(self) -> list[SimpleNamespace]:
                test.calls.append(("hardware",))
                return [
                    SimpleNamespace(name="t4-small", unit_cost_usd=0.006667, unit_label="minute")
                ]

            def __getattr__(self, name: str) -> None:
                raise AssertionError(f"Unexpected API operation: {name}")

        def factory(**kwargs: object) -> ReadOnlyApi:
            self.factory_options.append(kwargs)
            return ReadOnlyApi()

        self.factory = factory

    def args(self, *values: str) -> object:
        return module.parser().parse_args(values)

    def test_namespace_comes_from_job_url(self) -> None:
        self.assertEqual(
            module.job_target("https://huggingface.co/jobs/team/done", "other"), ("team", "done")
        )
        self.assertEqual(module.job_target("done", "account"), ("account", "done"))

    def test_ambiguous_or_foreign_job_targets_are_rejected(self) -> None:
        for value, owner in [
            ("done", None),
            ("http://huggingface.co/jobs/team/done", None),
            ("https://example.invalid/jobs/team/done", None),
            ("https://huggingface.co/jobs/team/done?token=private", None),
        ]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.job_target(value, owner)

    def test_failed_complete_and_running_jobs_remain_distinct(self) -> None:
        result = module.collect(
            self.args(
                "--job-owner", "account", "--job", "failed", "--job", "done", "--job", "live"
            ),
            self.factory,
        )
        self.assertEqual([j["stage"] for j in result["jobs"]], ["ERROR", "COMPLETED", "RUNNING"])
        self.assertEqual([j["terminal"] for j in result["jobs"]], [True, True, False])
        self.assertEqual([j["completed"] for j in result["jobs"]], [False, True, False])
        self.assertNotIn(SECRET, json.dumps(result))
        self.assertTrue(all(c[0] == "inspect" for c in self.calls))

    def test_public_model_uses_anonymous_metadata_without_weight_download(self) -> None:
        result = module.collect(
            self.args("--repo", "owner/model", "--public", "--revision", "a" * 40), self.factory
        )
        self.assertEqual(self.factory_options, [{"token": False}])
        self.assertEqual(result["model"]["files"][0]["lfs_sha256"], "b" * 64)
        self.assertTrue(all(c[0] == "model_info" for c in self.calls))
        self.assertNotIn("release_completed", result)

    def test_price_keeps_the_live_unit(self) -> None:
        result = module.collect(self.args("--hardware", "t4-small"), self.factory)
        self.assertEqual(result["hardware"]["unit_label"], "minute")
        self.assertEqual(result["hardware"]["unit_cost_usd"], 0.006667)

    def test_card_validation_does_not_instantiate_a_job_client(self) -> None:
        validations = []

        class Card:
            def __init__(self, text: str) -> None:
                self.text = text

            def validate(self, **kwargs: object) -> None:
                validations.append(kwargs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "README.md"
            path.write_text("---\nlanguage: en\n---\n# Fixture\n")
            result = module.collect(self.args("--card", str(path)), self.factory, Card)
        self.assertEqual(validations, [{"repo_type": "model"}])
        self.assertTrue(result["card"]["hub_metadata_valid"])
        self.assertEqual(self.factory_options, [])

    def test_sensitive_sdk_error_is_not_echoed(self) -> None:
        class FailingApi:
            def inspect_job(self, **kwargs: object) -> None:
                raise RuntimeError(SECRET)

        fake = SimpleNamespace(HfApi=FailingApi, ModelCard=object)
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict(sys.modules, {"huggingface_hub": fake}),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = module.main(["--job-owner", "account", "--job", "done"])
        self.assertEqual(code, 1)
        self.assertNotIn(SECRET, stdout.getvalue() + stderr.getvalue())
        self.assertEqual(json.loads(stderr.getvalue())["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
