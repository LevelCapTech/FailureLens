"""Register the earlier model versions in the MLflow registry (historical, not evaluated).

Reads the existing Stage 2 training records and imports each as a labelled historical
run plus a registry version. No training, no evaluation, no weights are copied. Runs
are skipped when the label is already registered, so the command can be repeated.

    python -m training.scripts.backfill_registry --provenance cloud-provenance.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.scripts.import_history import import_history

from failurelens.experiment_tracking import MLflowConfig
from failurelens.tracking_registry import REGISTERED_MODEL_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = Path("~/.local/share/failurelens").expanduser()


@dataclass(frozen=True)
class Entry:
    label: str
    run_name: str
    source: Path
    status: str


def default_entries(provenance: Path | None) -> list[Entry]:
    entries = [
        Entry(
            "v0.0.1-tainted-v1",
            "v0.0.1-tainted-v1: 学習・検証履歴(過去記録・使用禁止)",
            TRAIN_ROOT / "experiment-issue19-stage2" / "experiment_metadata.json",
            "tainted-do-not-use",
        ),
        Entry(
            "v0.0.2-tainted-v2",
            "v0.0.2-tainted-v2: 学習・検証履歴(過去記録・使用禁止)",
            TRAIN_ROOT / "experiment-issue19-stage2-v2" / "experiment_metadata.json",
            "tainted-do-not-use",
        ),
        Entry(
            "v0.1.0-local-v3",
            "v0.1.0-local-v3: ローカルv3 学習・検証履歴(過去記録)",
            TRAIN_ROOT / "experiment-issue19-stage2-v3" / "experiment_metadata.json",
            "experimental",
        ),
    ]
    if provenance is not None:
        entries.append(
            Entry(
                "v0.1.0",
                "v0.1.0: 公開モデル 学習・検証履歴(過去記録)",
                provenance,
                "experimental",
            )
        )
    return entries


def _client(uri: str) -> Any:
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def backfill(entries: list[Entry], uri: str, experiment: str, *, retire: bool) -> list[str]:
    client = _client(uri)
    registered: set[str] = set()
    if client.search_registered_models(filter_string=f"name = '{REGISTERED_MODEL_NAME}'"):
        registered = {
            v.tags.get("failurelens.version", "")
            for v in client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
        }
    messages: list[str] = []
    for entry in entries:
        if entry.label in registered:
            messages.append(f"skip {entry.label}: already registered")
            continue
        if not entry.source.exists():
            messages.append(f"skip {entry.label}: record not found")
            continue
        run_id = import_history(
            entry.source,
            tracking=MLflowConfig(
                tracking_uri=uri,
                experiment_name=experiment,
                run_name=entry.run_name,
                model_version=entry.label,
            ),
            status=entry.status,
        )
        messages.append(f"registered {entry.label} (run {run_id[:8]}, {entry.status})")
        if retire:
            digest = json.loads(entry.source.read_text(encoding="utf-8"))
            digest = digest.get("artifact_sha256")
            messages.extend(_retire_unlabelled(client, experiment, digest, keep=run_id))
    return messages


def _retire_unlabelled(client: Any, experiment: str, digest: str | None, keep: str) -> list[str]:
    """Soft-delete (recoverable) earlier unlabelled imports of the same record."""
    found = client.get_experiment_by_name(experiment)
    if found is None or not digest:
        return []
    messages: list[str] = []
    for run in client.search_runs([found.experiment_id]):
        if (
            run.info.run_id != keep
            and run.data.tags.get("failurelens.historical") == "true"
            and "model.version" not in run.data.tags
            and run.data.params.get("artifact_sha256") == digest
        ):
            client.delete_run(run.info.run_id)
            messages.append(f"retired unlabelled duplicate run {run.info.run_id[:8]}")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment", default="failurelens-development")
    parser.add_argument("--provenance", type=Path, help="Saved cloud failurelens_provenance.json.")
    parser.add_argument(
        "--retire-unlabelled-imports",
        action="store_true",
        help="Soft-delete earlier unlabelled imports of the same record (recoverable).",
    )
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    uri = args.tracking_uri or f"sqlite:///{REPO_ROOT / '.mlflow' / 'mlflow.db'}"
    for line in backfill(
        default_entries(args.provenance),
        uri,
        args.experiment,
        retire=args.retire_unlabelled_imports,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
