"""CLI entry point for the offline training smoke run."""

from __future__ import annotations

import argparse

from training.smoke import run_toy_training

from failurelens.experiment_tracking import add_tracking_arguments, tracking_from_args
from failurelens.notify import notify_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tiny offline training smoke loop")
    parser.add_argument("--config", default="training/configs/toy_config.json")
    parser.add_argument("--dataset", default="datasets/fixtures/sample_record.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    add_tracking_arguments(parser)
    args = parser.parse_args()
    tracking = tracking_from_args(parser, args)
    with notify_run("toy-training", args.mlflow_run_name) as notifier:
        result = run_toy_training(
            args.config,
            dataset_path=args.dataset,
            resume_from=args.resume_from,
            output_dir=args.output_dir,
            tracking=tracking,
        )
        notifier.set_metrics({"train.final_loss": result["metrics"]["final_loss"]})
    print(result["metrics"]["final_loss"])


if __name__ == "__main__":
    main()
