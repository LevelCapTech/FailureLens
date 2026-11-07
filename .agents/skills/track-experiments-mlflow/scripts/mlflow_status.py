"""Read-only summary of the local FailureLens MLflow store.

Usage: python .agents/skills/track-experiments-mlflow/scripts/mlflow_status.py [--uri URI]
Prints runs grouped by failurelens.kind and flags missing coverage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

ROOT = Path(__file__).resolve().parents[4]
KINDS = ("baseline-evaluation", "toy-training", "stage2-training")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=f"sqlite:///{ROOT / '.mlflow' / 'mlflow.db'}")
    args = parser.parse_args()
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=args.uri)
    exp = client.get_experiment_by_name("failurelens-development")
    if exp is None:
        print("No 'failurelens-development' experiment: nothing recorded yet.")
        return 1
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    seen: dict[str, int] = dict.fromkeys(KINDS, 0)
    for run in runs:
        tags = run.data.tags
        kind = tags.get("failurelens.kind", "?")
        seen[kind] = seen.get(kind, 0) + 1
        m = run.data.metrics
        keys = (
            "function_level.mrr",
            "function_level.top_1_accuracy",
            "train.loss",
            "validation.loss",
        )
        shown = {k: round(m[k], 4) for k in keys if k in m}
        print(
            f"{run.info.status:9} {kind:20} hist={tags.get('failurelens.historical', '-'):5} "
            f"dirty={tags.get('failurelens.git_dirty', '?'):7} {run.info.run_name} {shown}"
        )
    print()
    for kind, n in seen.items():
        print(f"{kind}: {n} run(s)" + ("  <-- none recorded" if n == 0 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
