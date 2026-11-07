"""CLI entry point for Stage 2 CodeRankEmbed fine-tuning.

Usage (from repo root, with venv activated and [ml] + [train] extras installed):

    python -m training.scripts.finetune_train \\
        --config training/configs/stage2_config.json \\
        --output-dir ~/.local/share/failurelens/experiment-issue19-stage2

    # Resume from epoch-2 checkpoint:
    python -m training.scripts.finetune_train \\
        --config training/configs/stage2_config.json \\
        --resume-from ~/.local/share/failurelens/experiment-issue19-stage2/checkpoint_epoch_2 \\
        --output-dir ~/.local/share/failurelens/experiment-issue19-stage2-resume

    # Point to a local model cache (avoids HF Hub download):
    python -m training.scripts.finetune_train \\
        --config training/configs/stage2_config.json \\
        --model-cache-dir /path/to/models/cache/nomic-coderankeembed-smoke

Prerequisites:
- GPU with CUDA (the script fails loudly if CUDA is unavailable).
- Private dataset at ~/.local/share/failurelens/train-issue17-v2/records.json
  with SHA256 74909f51...
- Model cache at <model-cache-dir>/models--nomic-ai--CodeRankEmbed/snapshots/
  3c4b60807d71f79b43f3c4363786d9493691f8b1/ (safetensors only, no .bin files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from failurelens.experiment_tracking import add_tracking_arguments, tracking_from_args
from failurelens.notify import RunNotifier, notify_run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 CodeRankEmbed fine-tuning")
    p.add_argument(
        "--config",
        default="training/configs/stage2_config.json",
        help="Hyperparameter config JSON",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for checkpoints and final artifact",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint directory to resume from (e.g. .../checkpoint_epoch_2)",
    )
    p.add_argument(
        "--model-cache-dir",
        default=None,
        help="Local HF model cache folder (optional; avoids network calls)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Override private dataset root (default: ~/.local/share/failurelens/train-issue17-v2)",
    )
    add_tracking_arguments(p)
    args = p.parse_args()
    args.tracking = tracking_from_args(p, args)
    return args


def main() -> None:
    args = _parse_args()
    run_name = args.tracking.run_name if args.tracking is not None else None
    with notify_run("stage2-training", run_name) as notifier:
        _run(args, notifier)


def _run(args: argparse.Namespace, notifier: RunNotifier) -> None:

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config: dict = json.loads(config_path.read_text(encoding="utf-8"))

    data_root = Path(args.data_root) if args.data_root else None
    print("Loading Stage 2 training examples...")
    if config.get("corpus") == "failurelens-corpus-v02":
        # v0.2 private corpus: repo-level train/dev split, pinned by combined digest.
        from training.corpus.loader import load_v02_examples

        train_examples, val_examples = load_v02_examples(
            data_root, expected_sha256=str(config.get("dataset_sha256", ""))
        )
    else:
        from training.data_stage2 import load_stage2_examples

        train_examples, val_examples = load_stage2_examples(data_root)
    print(f"  train={len(train_examples)} val={len(val_examples)}")

    # Brief content reconstruction report (aggregate, no individual labels)
    for split_name, examples in [("train", train_examples), ("val", val_examples)]:
        repos = sorted({ex.source_repository for ex in examples})
        avg_pos_len = sum(len(ex.positive) for ex in examples) // max(1, len(examples))
        avg_neg_len = sum(len(ex.negative) for ex in examples) // max(1, len(examples))
        print(
            f"  {split_name}: repos={repos} avg_pos_chars={avg_pos_len} avg_neg_chars={avg_neg_len}"
        )

    from training.finetune import run_stage2_finetune

    output_dir = Path(args.output_dir) if args.output_dir else None
    resume_from = Path(args.resume_from) if args.resume_from else None

    print(f"\nStarting Stage 2 fine-tuning (epochs={config.get('epochs', 3)})...")
    if resume_from:
        print(f"  Resuming from: {resume_from}")

    result = run_stage2_finetune(
        config,
        train_examples=train_examples,
        val_examples=val_examples,
        output_dir=output_dir,
        resume_from=resume_from,
        model_cache_dir=args.model_cache_dir,
        tracking=args.tracking,
    )

    history = result["history"]
    notifier.set_metrics(
        {
            "train.loss": history["train_loss"][-1],
            "validation.loss": history["val_loss"][-1],
            "train.elapsed_seconds": result["elapsed_seconds"],
            "train.peak_gpu_memory_mib": result["peak_gpu_memory_mib"],
        }
    )

    print("\n=== Experiment complete ===")
    print(f"Git SHA:           {result['git_sha']}")
    print(f"Trainable params:  {result['trainable_params']:,} / {result['total_params']:,}")
    print(f"Artifact:          {result['artifact_path']}")
    print(f"Artifact SHA256:   {result['artifact_sha256']}")
    print(f"Elapsed:           {result['elapsed_seconds']:.1f}s")
    print(f"Peak GPU memory:   {result['peak_gpu_memory_mib']} MiB")
    print(f"Train loss history: {result['history']['train_loss']}")
    print(f"Val loss history:   {result['history']['val_loss']}")


if __name__ == "__main__":
    main()
