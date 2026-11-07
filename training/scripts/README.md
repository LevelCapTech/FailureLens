# Training scripts

Entry points for training/fine-tuning, plus a **CPU-only tiny smoke test** that
exercises the pipeline end to end without a GPU (required by the project issue).

Scripts must depend only on the optional `train` extra, record full run metadata
for reproducibility, and support checkpoint resume.

> Status: **experimental implementation.** `smoke_train.py` runs the CPU toy
> pipeline; `finetune_train.py` runs Stage 2 GPU fine-tuning with private,
> digest-checked inputs. Independent model-quality evaluation remains pending.

`import_history.py` imports an existing Stage 2 metadata or cloud provenance JSON
into MLflow, preserving actual loss histories and artifact identity. It does not
train, evaluate accuracy, or upload weights. See the
[experiment tracking guide](../../docs/experiment-tracking.md).
