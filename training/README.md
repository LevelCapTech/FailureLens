# Training

Experimental training pipelines for the FailureLens fault-localization model.

> **Status: experimental implementation.** A CPU-only toy pipeline and Stage 2
> GPU fine-tuning exist. A Stage 2 cloud rebuild is published at
> [LevelCapTech/failurelens](https://huggingface.co/LevelCapTech/failurelens);
> independent held-out evaluation remains pending. Heavy training depends on
> optional ML dependencies, runs only when explicitly invoked on a local or
> cloud GPU, and never runs in CI or as part of the default install.

## Layout

| Path | Purpose | Status |
|------|---------|--------|
| `configs/` | Toy and Stage 2 experiment configs (base pin, dataset, hyperparameters, seed). | Experimental implementation |
| `scripts/` | CPU toy smoke and Stage 2 GPU fine-tuning entry points. | Experimental implementation |
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
For the bounded cloud rebuild, transfer approval and publication checks, see
[`docs/hf-jobs-publication.md`](../docs/hf-jobs-publication.md).
