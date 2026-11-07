# Development experiment tracking with MLflow

**Working, optional:** the baseline evaluation harness, CPU toy training loop,
and experimental Stage 2 GPU loop can record measured results in MLflow. Tracking
is disabled unless `--mlflow` or an explicit `MLflowConfig` is supplied. Core
diagnosis remains offline and requires only the existing core dependencies.

This integration does not establish model accuracy or run independent held-out
evaluation. Toy training loss is a pipeline check, not a fault-localization score.

## Local setup

Run from the repository root with Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
make tracking-install
make eval-track
make train-smoke-track
make mlflow-ui
```

Open <http://127.0.0.1:5000> and select `failurelens-development`. The UI binds
only to loopback. Change its port with `make mlflow-ui MLFLOW_PORT=5001`.
Evaluation and training can write runs before the UI starts.

Run metadata lives in `.mlflow/mlflow.db` (SQLite); artifacts live in
`.mlflow/artifacts/`. The location is resolved from the repository root, so
changing the current directory does not select a different default store. All
tracking state and generated toy checkpoints are ignored by Git. `make clean`
preserves experiment history. Back up the whole `.mlflow/` directory with writers
and the UI stopped; artifact URIs are absolute, so restoring at a different path
requires updating artifact locations.

The tracking extra includes the local MLflow server/UI and does not install
torch or transformers. MLflow telemetry is disabled by default for these runs
and by the UI command (`MLFLOW_DISABLE_TELEMETRY=true`).

## Compare development changes

Record a run before and after a code change, using the same benchmark:

```bash
python evaluation/run_baseline_eval.py --mlflow --mlflow-run-name before-change
# Make a code change, then repeat the evaluation.
python evaluation/run_baseline_eval.py --mlflow --mlflow-run-name after-change
```

Select the two runs in the experiment's runs table and use **Compare**. Filter
by the `failurelens.kind` tag to distinguish `baseline-evaluation`, `toy-training`,
and `stage2-training`. Use these columns/charts:

| Metric or metadata | Interpretation |
| --- | --- |
| `file_level.mrr`, `function_level.mrr` | Ranking quality; larger is better. |
| `file_level.top_1_accuracy`, `function_level.top_1_accuracy` | Fraction with the first prediction relevant. |
| `file_level.recall_at_5`, `function_level.recall_at_5` | Ground-truth locations recovered among the first five. |
| `file_level.num_cases`, `function_level.num_cases` | Actual denominators; function metrics require function labels. |
| `performance.diagnosis_mean_ms`, `performance.diagnosis_max_ms` | Wall time per case, including indexing and ranking, excluding MLflow logging. |
| `train.loss`, `validation.loss` | Loss curves by epoch; validation loss exists only for Stage 2. |
| `train.val_gap` | Validation minus training loss per epoch. Near zero is healthy; rising suggests overfitting. |
| `train.learning_rate` | Learning rate after each epoch (measured for new runs; derived from the known cosine schedule for imported history, marked in its parameters). |
| `diag.*` | Overfitting indicators, see below. |
| `model.version` (tag and parameter) | Release label such as `v0.2.0-candidate-1`; set with `--model-version`. |
| `dataset_sha256` | Identity of the evaluation or training data. |
| `train.peak_gpu_memory_mib`, `train.elapsed_seconds` | GPU training resource measurements. |
| `benchmark_sha256`, `dataset_sha256` | Input identity for comparable experiments. |
| `mlflow.source.git.commit`, `failurelens.git_dirty` | Code revision and whether uncommitted changes were present. |
| `run.duration_seconds` | Tracked scope duration, including logging overhead; not inference latency. |

The bundled benchmark is a single demonstration case. Its results are useful
for regressions, not evidence of generalization. The current harness returns at
most five candidates (`candidate_limit`), even when reporting metric keys with
larger k. Compare like-for-like inputs, candidate limits, hardware and dependency
versions. Timings are single observations per case, not a controlled performance
benchmark. Function metrics are omitted when no function labels exist.

Each evaluation records `evaluation.json` and `environment.json`. The latter
contains Python/platform architecture and installed versions of the relevant
FailureLens, MLflow and ML dependencies. It does not capture arbitrary environment
variables, a full package inventory, a source diff, or user information.

## Track training

```bash
python -m training.scripts.smoke_train \
  --mlflow --mlflow-run-name cpu-smoke

# Only with the separately prepared GPU environment and private training data:
python -m training.scripts.finetune_train \
  --config training/configs/stage2_config.json \
  --mlflow --mlflow-run-name stage2-experiment
```

The toy loop records its effective simple gradient-descent optimizer and batch
size rather than claiming to implement all settings in its experimental config.
Stage 2 records its effective optimizer settings, pinned model identity, dataset
digest, example/parameter counts and GPU metadata. Both save a filtered
`config.json` and aggregate `training.json`. They never upload dataset records,
queries, source code or private file paths. Model weights are never sent to a
remote store; see [Model Registry](#model-registry-and-comparing-versions).

`--resume-from` starts a **new** MLflow run with `resumed=true` and `resume_epoch`.
New losses retain their actual epoch numbers. Previous epoch losses remain in
the aggregate history artifact, and are not replayed as new measurements. The
existing checkpoint/resume behavior is unchanged. Tracking uses explicit run IDs
and leaves any caller-owned fluent MLflow active run and tracking URI untouched.

Completed runs become `FINISHED`; exceptions and interruptions mark an already
created run `FAILED`. An unavailable explicitly selected tracking store fails the
experiment rather than silently losing measurements. If it prevents recording
the failure status, the original experiment exception is preserved and a warning
is emitted. Failures before a run can be created cannot be stored in MLflow.

## Model Registry and comparing versions

Give a run a release label with `--model-version` (`v0.1.0`, `v0.2.0-candidate-1`; the
`v` and three numbers are required). Stage 2 training and `import_history` then
register a version of the MLflow registered model `failurelens`. Each registry
version carries the tags `failurelens.version`, `source_commit`, `weight_sha256`,
`evaluation_status` (starts as `not evaluated`; other values are `gate-c-pending`,
`evaluated-development`, `gate-c-evaluated`), `status` (`experimental` or
`tainted-do-not-use`), `historical` and `weights_in_store`.

Add measured aggregate scores later to the **same** version:

```python
from failurelens.tracking_registry import record_evaluation

record_evaluation(
    "v0.2.0-candidate-1", {"function_level.mrr": 0.55, "function_level.top_1_accuracy": 0.41}
)
```

They become `eval.<name>` tags on the version and `eval.<name>` metrics on its training run.
Only finite numbers with simple names are accepted. The status defaults to
`evaluated-development`; `gate-c-evaluated` is reserved for an approved independent
evaluation, and recording scores is not itself a Gate C decision.

**Weights.** When the tracking store is local (SQLite or file backend with file
artifacts), a trained checkpoint is copied into that store as the version's artifact so
the version is self-contained. For a remote or non-file store (an HTTP server, cloud
storage) weights are **never uploaded**; only the SHA-256 and a hash of the local path
are recorded in `model_metadata/weights.json`. `import_history` has no weights to copy
and records the digest only.

**Overfitting indicators** (`diag.*`, from the epoch curves; epochs are 1-based):

| Metric | Meaning |
| --- | --- |
| `diag.best_epoch`, `diag.best_validation_loss` | Epoch with the lowest validation loss. |
| `diag.epochs_since_best` | Epochs after that best one. |
| `diag.final_gap` | Validation minus training loss at the last epoch. |
| `diag.gap_trend` | Least-squares slope of the gap per epoch; positive means widening. |
| `diag.overfit_signal` | 1.0 only if validation is worse than its best while training loss kept improving. A heuristic warning, not proof: curves from few epochs and few examples are noisy. |

**Compare versions in graphs.**

```bash
make compare                 # writes .mlflow/reports/index.html and four PNG graphs
```

`index.html` explains the graphs in plain words and lists each version with its final
losses, best epoch, overfitting signal, evaluation scores and evaluation status
(`not evaluated` until real scores exist). Tainted versions are hidden; add
`--include-tainted` to show them. In the MLflow UI (`make mlflow-ui`) you can also open
**Models**, select `failurelens`, and compare versions; for run-level curves select
several runs in the `failurelens-development` runs table, press **Compare**, and choose
the line chart for `train.loss`, `validation.loss`, `train.val_gap` or
`train.learning_rate`. Filter by `tags.model.version` to see one version.

Training runs, evaluations and history imports also send a start/finish/fail
notification; see [notifications](notifications.md) for the one-time setup.

## Import existing Stage 2 training records

Earlier experiments were recorded as JSON before MLflow was introduced. Their
loss curves do not appear automatically in the tracking UI. Import an existing
`experiment_metadata.json`, or a locally saved `failurelens_provenance.json`
from the cloud release, without retraining or transferring model weights:

```bash
python -m training.scripts.import_history /path/to/experiment_metadata.json \
  --mlflow --mlflow-run-name historical-stage2

# Also register it as a model version (no weights are copied), for example:
#   --model-version v0.1.0 [--status tainted-do-not-use]
```

The new run contains the original epoch-wise `train.loss` and `validation.loss`,
the measured training time/memory, the original source revision and weight digest,
and a `loss_curves.png` plot. `failurelens.historical=true` and
`measurement_scope=historical-training-only` distinguish an import from a new
training run. The UI timestamp is the import time, not an invented training
timestamp. `import_environment.json` describes the importer;
`training_environment.json` contains only available original environment fields.
The import duration is recorded separately from the original training duration.

Only selected aggregate fields are stored; original JSON with private paths,
records and weights is not uploaded. Invalid or non-finite loss histories,
inconsistent epoch counts and mismatched cloud artifact digests are rejected.
Each invocation creates a new explicitly labelled historical run.
The historical local v3 and the published cloud rebuild have different weight
digests and must remain separate runs. Importing loss curves does not perform
independent model-quality evaluation or establish accuracy.

## Configuration and shared servers

The evaluation, training and history-import entry points accept:

```text
--mlflow
--mlflow-tracking-uri URI
--mlflow-experiment NAME
--mlflow-run-name NAME
```

URI precedence is explicit flag, `MLFLOW_TRACKING_URI`, then the repository's
local SQLite store. Experiment precedence is explicit flag,
`MLFLOW_EXPERIMENT_NAME`, then `failurelens-development`. Environment variables
alone do not enable tracking. Extra tracking flags require `--mlflow`.

To use an already configured team tracking server, explicitly select it:

```bash
MLFLOW_TRACKING_URI=https://mlflow.example.org \
MLFLOW_EXPERIMENT_NAME=failurelens-development \
python evaluation/run_baseline_eval.py --mlflow
```

This sends the documented metadata and aggregate artifacts to that server. Use
the server's authentication environment variables; do not put credentials in
commands, configs, URI query strings or repository files. The local UI target
always opens the local database; use the shared server's own UI for remote runs.
Remote deployment, authentication provisioning and artifact storage are outside
this local setup.

Python callers can pass the same explicit config to `run`, `run_toy_training`,
or `run_stage2_finetune`:

```python
from pathlib import Path
from evaluation.run_baseline_eval import run
from failurelens.experiment_tracking import MLflowConfig

report = run(
    Path("evaluation/benchmarks/sample_cases.json"),
    tracking=MLflowConfig(run_name="baseline-check"),
)
```

## Verification

`make check` includes tests for the disabled path, lifecycle, configuration and
dependency errors. With the tracking extra installed it also exercises real
SQLite runs, artifact round trips, actual baseline metrics, toy loss curves and
resume steps. CI has a separate CPU tracking job so the core test matrix keeps
its small dependency set. GPU integration tests remain separately gated and do
not run in CI.

The backend and explicit-run client API follow the official
[MLflow tracking server documentation](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
and [MLflow client reference](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).
