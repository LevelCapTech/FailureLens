"""CLI entry point for the offline training smoke run."""

from __future__ import annotations

import argparse

from training.smoke import run_toy_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tiny offline training smoke loop")
    parser.add_argument("--config", default="training/configs/toy_config.json")
    parser.add_argument("--dataset", default="datasets/fixtures/sample_record.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()
    result = run_toy_training(
        args.config,
        dataset_path=args.dataset,
        resume_from=args.resume_from,
        output_dir=args.output_dir,
    )
    print(result["metrics"]["final_loss"])


if __name__ == "__main__":
    main()
