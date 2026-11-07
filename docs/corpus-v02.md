# v0.2 private fault corpus (train and dev)

**Status: generator, loader, manifests and candidate configs are implemented and
tested on a synthetic project. The corpus itself has not been generated yet, so
there are no record counts, no validation numbers and no model results.** Do not
cite numbers from this page; the build prints its own aggregate report.

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

The generator executes third-party test code. Run it only on the pinned,
license-reviewed repositories, in the dedicated virtualenv, from a trusted shell:

```bash
ROOT=~/.local/share/failurelens/v02-corpus
PYTHONPATH=packages/python/src:. python -m training.corpus.build prepare  --root $ROOT --python $ROOT/venv/bin/python
PYTHONPATH=packages/python/src:. python -m training.corpus.build generate --root $ROOT --python $ROOT/venv/bin/python
PYTHONPATH=packages/python/src:. python -m training.corpus.build finalize --root $ROOT --python $ROOT/venv/bin/python
```

`finalize` prints an aggregate report and the **corpus digest**. Put the digest
into `dataset_sha256` of `training/configs/stage2_v02_*.json` (committed configs
carry the placeholder `PENDING_CORPUS_BUILD`, which the loader refuses).
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
