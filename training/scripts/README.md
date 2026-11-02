# Training scripts

Entry points for training/fine-tuning, plus a **CPU-only tiny smoke test** that
exercises the pipeline end to end without a GPU (required by the project issue).

Scripts must depend only on the optional `train` extra, record full run metadata
for reproducibility, and support checkpoint resume.

> Status: **experimental implementation.** `smoke_train.py` runs the CPU toy
> pipeline; `finetune_train.py` runs Stage 2 GPU fine-tuning with private,
> digest-checked inputs. Independent model-quality evaluation remains pending.
