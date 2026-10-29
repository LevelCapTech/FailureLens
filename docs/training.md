# Training

Specification for the FailureLens training pipeline. This repo now includes a
small, CPU-only toy pipeline used to validate the training contract without
network access or pretrained weights. It is intentionally limited to local
fixture data and is not a production model release.

## Scope and task

The model's job is **retrieval/ranking** — estimating how related an error is to
a candidate code unit — **not** text generation. The first model is planned as a
small **bi-encoder** fine-tuned for fault localization, chosen so it can:

- fine-tune on a single local GPU,
- run inference on CPU (and faster on GPU),
- avoid external APIs,
- be distributed as pinned weights on the Hugging Face Hub.

Concrete parameter counts, latencies, and accuracy targets are decided by
**experiment**, not asserted up front.

## Base model

Candidate: [`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed)
(~137M-param code-search bi-encoder, MIT per its model card). Before adoption,
verify: license, architecture, inference method, preprocessing, token length,
fine-tuning feasibility, dependency code, **pinned revision**, and security
requirements. If loading requires custom code, review the exact code at a pinned
revision (see `docs/security.md`).

## Objectives

Research candidates (pick **one** first and build a reproducible experiment):
contrastive learning, pairwise ranking, triplet loss, InfoNCE, hard-negative
mining. Implementations live under `training/objectives/`.

## Positive / negative labels

- **Query**: error log + structured failure info.
- **Positive**: code confirmed as the cause.
- **Negative**: code that is not the cause.

Labels must come from trustworthy sources. An injected-bug location is **not**
automatically the root cause; define label semantics explicitly (see
`docs/datasets.md`).

## Pipeline capabilities (§11)

Config-driven: dataset loading, preprocessing, tokenisation, model loading, train
loop, validation, checkpointing + **resume**, metric logging, history, and final
export.

Configurable keys (see `training/configs/README.md`): `base_model`, `revision`,
`dataset`, `dataset_version`, `batch_size`, `learning_rate`, `epochs`,
`max_length`, `seed`, `optimizer`, `objective`, `mixed_precision`, `output_dir`,
`val_interval`, `early_stopping`.

## Reproducibility

Every experiment records: git commit, dataset + version, config file, random
seed, dependency versions, GPU info, and results (`training/experiments/`). A
**CPU-only, tiny** training smoke test must exist so the pipeline is testable
without a GPU.

## Environment (§12)

- Candidate GPUs: RTX 3060 / 5060 Ti / 5070, Colab Pro. **Do not** infer VRAM
  from a model number — detect it at runtime.
- Emit a clear error when CUDA is unavailable; **do not** silently switch regimes.
- Keep data generation and GPU training as separate steps.
- Keep training and inference dependencies separate (`train` vs `ml` extras).
- **Never** run large-scale generation or training in CI.

## Status checklist

- [ ] CPU-only tiny smoke test
- [ ] Config schema + loader
- [ ] Dataset loader + preprocessing (reuse library parsers/indexers)
- [ ] One objective implemented
- [ ] Checkpoint + resume
- [ ] Experiment metadata recording
- [ ] Model export + model card

Nothing above is implemented yet; this document is the plan of record.
