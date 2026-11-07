# v0.2 private fault corpus (train and dev)

**Status: the corpus was generated on 2026-11-07 inside a network-less Docker
sandbox and passed the automated gates (178 train and 62 dev records; counts
below). No model has been trained on it yet and there are no model results.**
Counts are aggregate only; record content stays in the private directory.

This page describes how the v0.2 training set and a separate development
(model-selection) set are produced. It does not authorize training, release or
the Gate C run. It is not a public dataset: records, clones, failing trees and
manifests stay in a private directory outside Git.

## Why a second set

The Gate C hold-out (`psf/requests`, `pallets/werkzeug`, `pypa/pip`) must only
be used for the final independent measurement. v0.2 candidates therefore need
their own pass/fail signal. The **dev set** is built from different
repositories, measured with the same evaluator path as Gate C
(`evaluation/independent_runner.py` with `evaluation/product_pool.py`), and used
for validation loss and choosing between candidate runs.

| Split | Repositories (pinned in `training/corpus/repos.py`) | Use |
| --- | --- | --- |
| train | `pallets/click`, `more-itertools/more-itertools`, `pytoolz/toolz`, `astanin/python-tabulate`, `kjd/idna`, `jpvanhal/inflection`, `pallets/itsdangerous` (plus the earlier `python-attrs/attrs`, not regenerated) | fine-tuning |
| dev | `andialbrecht/sqlparse`, `python-humanize/humanize`, `mahmoud/boltons`, `jmespath/jmespath.py` | validation, model selection, dev evaluation |
| hold-out | `psf/requests`, `pallets/werkzeug`, `pypa/pip` | Gate C only, never read here |

Splits are by repository. `validate_table` and `finalize` fail if a repository
appears in more than one split or in the hold-out. License evidence and pins:
`docs/gate-b-rights-ledger.md`.

## How a record is made

1. Pick a module-level, uniquely named function that at least one passing test
   calls (call sets come from a pytest plugin, `training/corpus/trace_plugin.py`).
2. Apply **one exact single-line mutation** (`training/corpus/mutations.py`):
   comparison, boolean or arithmetic operator flip, constant off-by-one, boolean
   constant flip, negation removal, wrong-value return, early `return None`, or an
   inserted exception. The recipe is a whole-line `old`/`new` pair, the form
   `evaluation.product_pool.verify_failing_tree` checks.
3. Run the covering tests with the project's own pytest. Keep the case only if
   the failing test passes on the clean tree, fails after the change (exit 1,
   native Python traceback) and the work copy is restored byte for byte.
4. Label the changed function positive and one executed and one never-executed
   unmodified function negative. The rationale separates the surfacing test from
   the root cause.
5. Paths in the log are made repository-relative. Then the JSON schema, the
   semantic gates (`failurelens.quality`: license, candidates, rationale,
   pass-before/fail-after, environment signatures, secret/PII screens,
   duplicates) and a duplicate-edit check run. Failures are dropped and counted.

Dev cases additionally get a failing tree per case and the evaluator manifests
(`source_manifest.json`, `tree_manifest.json`, `train_manifest.json`) in the
shapes `load_private_cases` and `load_product_cases` expect. The synthetic-project
test loads them through both loaders.

## Running it (opt-in, never in CI)

The generator executes third-party test code, so it never runs on the host.
`training/corpus/sandbox/sandbox_python.sh` is a drop-in `--python` that runs the
interpreter in a container with: `--network none`, the host (non-root) uid, a
read-only root file system, all capabilities dropped, `no-new-privileges`, 3 GiB
memory / 2 CPU / 512 pids limits, a 60 s per-test timeout (`pytest-timeout`) and a
hard 1500 s in-container timeout. Only the private work copy of one repository, the
`PYTHONPATH` directories (read-only) and the trace output directory are mounted;
`$HOME`, `~/.ssh`, `~/.config` and tokens are not. The image is built from a
digest-pinned `python:3.12-slim` with hash-pinned wheels
(`training/corpus/sandbox/requirements-sandbox.txt`); the build is the only
network-enabled step. The trace plugin emulates what an editable install would
provide (a generated `_version` module and `importlib.metadata.version`) in
memory, so the checkouts are not modified.

```bash
docker build -t failurelens-corpus-sandbox:v02 training/corpus/sandbox
ROOT=~/.local/share/failurelens/v02-corpus
SANDBOX=$PWD/training/corpus/sandbox/sandbox_python.sh
PYTHONPATH=packages/python/src:. python -m training.corpus.build prepare  --root $ROOT --python $SANDBOX
PYTHONPATH=packages/python/src:. python -m training.corpus.build generate --root $ROOT --python $SANDBOX \
  --train-target 40 --dev-target 20 --jobs 3
PYTHONPATH=packages/python/src:. python -m training.corpus.build finalize --root $ROOT --python $SANDBOX
```

`generate` is resumable per repository (a finished repository is skipped unless
`--force`). One repository's very large parametrized file is excluded in
`training/corpus/repos.py` (`click`: `test_stream_lifecycle.py`, about 30k cases).

## Result of the 2026-11-07 build (aggregate)

Corpus digest (`corpus_sha256`):
`fd229415d23cbf8a0872a480746a3b11a576d120886dfefc056bf0c66ab48255`.
Train overlap with dev: 0 repositories. Overlap with the Gate C hold-out: 0.

| Split | Kept | Input | Dropped by the gates |
| --- | ---: | ---: | --- |
| train | 178 | 196 | 12 duplicate, 9 PII-pattern |
| dev | 62 | 70 | 6 duplicate, 1 environment failure, 1 secret-pattern |

| Repository | Split | Records |
| --- | --- | ---: |
| pytoolz/toolz | train | 39 |
| pallets/click | train | 38 |
| kjd/idna | train | 33 |
| more-itertools/more-itertools | train | 26 |
| astanin/python-tabulate | train | 16 |
| jpvanhal/inflection | train | 14 |
| pallets/itsdangerous | train | 12 |
| python-humanize/humanize | dev | 19 |
| mahmoud/boltons | dev | 18 |
| andialbrecht/sqlparse | dev | 16 |
| jmespath/jmespath.py | dev | 9 |

Mutation kinds (train / dev): raised exception 38 / 14, early-return None 35 / 9,
wrong-value return 32 / 11, comparison-operator flip 19 / 7, constant off-by-one
17 / 4, negation removal 14 / 5, arithmetic-operator flip 11 / 5, boolean-operator
flip 7 / 5, boolean constant flip 5 / 2.

Candidate mutations that no test detected were dropped rather than kept (for
example 32 of 72 attempts for `click` and 26 of 54 for `more-itertools`), and 17
`humanize` attempts were rejected because the failure log had no usable Python
traceback. `python-attrs/attrs` was not regenerated in this build and contributes no v0.2 records.

The configs `training/configs/stage2_v02_{a,b,c}.json` carry this digest in
`dataset_sha256`; the loader refuses a corpus whose digest differs.

`finetune_train.py` selects the v0.2 loader when a config has
`"corpus": "failurelens-corpus-v02"`. Train records become training examples; dev
records become the validation examples.

`APPROVAL_ID` in `training/corpus/build.py` records that the manifest was
produced under the owner's delegation for the v0.2 cycle; it is a label, not an
independent approval of the sources or labels.

## Limitations (stated up front)

- Synthetic single-line mutations of module-level functions only; methods are
  excluded so function labels stay unambiguous for the product indexer. Natural
  defects differ.
- Labels follow from construction (one sole edit) and passing tests. Tests that
  do not exercise a mutation leave it undetected and it is dropped, so the corpus
  skews toward well-tested code.
- Two negatives per record at most; the "executed" negative is not guaranteed to
  be hard for a given model.
- Automated gates are screeners, not independent human label review, and are not
  evidence of model quality.
