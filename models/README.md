# Models

Model **configurations** and **architecture** definitions for FailureLens.

> **Status.** A trained experimental encoder is published at
> [LevelCapTech/failurelens](https://huggingface.co/LevelCapTech/failurelens).
> Independent held-out evaluation remains pending; no superiority claim is made.
> The installable library's offline baseline remains available. See
> `docs/model-card.md` for the distinct local v3 and cloud-release artifacts,
> and the Hub model card for the supported standalone inference example.

This directory is for *source-controlled* model definitions and configs. Trained
weights are **not** stored in git - they are distributed separately on the
Hugging Face Hub and loaded on demand (never forced).

## Layout

| Path | Purpose | Status |
|------|---------|--------|
| `configs/` | YAML/JSON model configuration files (base model id, revision, pooling, max length, ...). | Planned |
| `architectures/` | Python modules defining model heads / wrappers around a base encoder. | Planned |

## Intended direction

The task is **retrieval / ranking**. The experimental Stage 2 model is a small
**bi-encoder** fine-tuned for fault localization.

The experimental base model has a pinned, audited revision; independent task
quality and adoption remain unverified:

- [`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed) - a
  code-retrieval bi-encoder (~137M params, MIT), pinned at
  `3c4b60807d71f79b43f3c4363786d9493691f8b1`.

When a config references a base model, it **must** pin a `revision`. If loading a
base model would execute custom code, the exact code and pinned revision must be
reviewed first (see `docs/security.md`).

## Model license

The GitHub source is Apache-2.0, but a trained model's license depends on its
**base model and training data**. Do not assume Apache-2.0 applies to released
weights; determine the correct license from the base model and datasets before
publishing anything. See `docs/model-card.md`.
