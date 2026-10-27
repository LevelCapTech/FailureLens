# Training

Training pipeline for the (future) FailureLens fault-localization model.

> **Status: design + scaffolding.** No training code or trained weights exist
> yet, and the library works today via the dependency-light baseline. Training
> depends on the optional `train` extra and is intended to run on a local GPU or
> Colab - **never** in CI, and never as part of the default install.

## Layout

| Path | Purpose | Status |
|------|---------|--------|
| `configs/` | Experiment configuration files (base model, dataset, hyperparameters, seed). | Planned |
| `scripts/` | Entry points for training / fine-tuning and a CPU-only smoke test. | Planned |
| `objectives/` | Loss / objective implementations (e.g. contrastive, pairwise ranking). | Planned |
| `experiments/` | Recorded experiment metadata and outputs (not weights). | Generated |

## Intended capabilities (issue §11)

Config-driven training with: dataset loading, preprocessing, tokenisation, model
loading, train loop, validation, checkpointing + resume, metric logging, history,
and final export. Configurable: base model + **revision**, dataset + version,
batch size, learning rate, epochs, max length, seed, optimizer, objective, mixed
precision, save path, validation interval, early stopping.

## Reproducibility (issue §11)

Every experiment must record: git commit, dataset + version, config file, random
seed, dependency versions, GPU info, and results. A **CPU-only, tiny** training
smoke test must exist so the pipeline is testable without a GPU. If a GPU is
unavailable, fail loudly - do not silently change the training regime.

## Environment

- Do not infer VRAM solely from a GPU model number; detect it at runtime.
- Emit a clear error when CUDA is unavailable.
- Keep data generation and GPU training as separate steps.
- Keep training and inference dependencies separate.

See `docs/training.md` for the full specification.
