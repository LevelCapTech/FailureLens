# Models

Model **configurations** and **architecture** definitions for FailureLens.

> **Important honesty note.** There is currently **no trained FailureLens model**.
> This directory describes the intended design and holds configs/architecture
> code as it is developed. The installable library works today using a
> dependency-light **baseline** (stack-trace + lexical); see the repository
> `README.md` and `docs/model-card.md`.

This directory is for *source-controlled* model definitions and configs. Trained
weights are **not** stored in git - they are distributed separately on the
Hugging Face Hub once they exist, and loaded on demand (never forced).

## Layout

| Path | Purpose | Status |
|------|---------|--------|
| `configs/` | YAML/JSON model configuration files (base model id, revision, pooling, max length, ...). | Planned |
| `architectures/` | Python modules defining model heads / wrappers around a base encoder. | Planned |

## Intended direction

The task is **retrieval / ranking**, not text generation, so the first model is
planned as a small **bi-encoder** fine-tuned for fault localization rather than a
large generative LLM.

Candidate base model (to be validated before adoption - license, architecture,
revision, custom-code requirements, token limits, fine-tuning feasibility):

- [`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed) - a
  code-retrieval bi-encoder (~137M params, MIT per its Hugging Face page).

When a config references a base model, it **must** pin a `revision`. If loading a
base model would execute custom code, the exact code and pinned revision must be
reviewed first (see `docs/security.md`).

## Model license

The GitHub source is Apache-2.0, but a trained model's license depends on its
**base model and training data**. Do not assume Apache-2.0 applies to released
weights; determine the correct license from the base model and datasets before
publishing anything. See `docs/model-card.md`.
