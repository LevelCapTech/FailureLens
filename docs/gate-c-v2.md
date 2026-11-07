# Gate C v2: a fresh final-evaluation set

**Status: built and structurally checked on 2026-11-07. No model has been run on
it and there are no accuracy results.** This page describes the set in
aggregate only. Records, logs, source snapshots, failing trees, labels and
per-case metadata are private, live outside Git, and must never be committed or
quoted. It is a measurement set, not a Gate C decision, and it authorizes
nothing: training, publication and adoption need separate approval.

## Why a second hold-out

The first hold-out (`psf/requests`, `pallets/werkzeug`, `pypa/pip`) was built
before the shipped retrieval was wired to it. Its error logs name files that do
not exist in the repository trees, so the shipped baseline returned an empty
pool for 29 of 30 cases and the shipped-baseline comparison could not separate
any model from any other (see `docs/releases/v0.2/REPORT.md`). Aggregate
numbers from it have also already been looked at once per measured candidate.
Gate C v2 fixes both problems with new repositories and logs that carry real,
repository-relative paths.

## What it contains (aggregate)

| Item | Value |
| --- | --- |
| Cases | 40, all function-labelled |
| Repositories | 4 (`tkem/cachetools`, `PyCQA/pycodestyle`, `python-validators/validators`, `bottlepy/bottle`), 10 cases each |
| Licenses | MIT for all four; pins and license-blob digests in `docs/gate-b-rights-ledger.md` |
| Records SHA-256 | `87a4676314bc24f0db10c020ce7953223d5c6953d20a08a174b4a5d1021a83e6` |
| Source / tree / train manifest SHA-256 | `47bf20fb...c1e24` / `e7f6d69d...f09a` / `f7cef059...afd0` (train manifest is regenerated per training list, see below) |
| Generator | `training/corpus/gatec_v2.py` (`gatec-v2-1.0`), seed `20261108` |
| Third-party code execution | only inside the network-less Docker sandbox of `training/corpus/sandbox/` |

Every case passes the schema and the semantic gates, the clean tree passes the
failing test, the edited tree fails it (exit code 1, native traceback), and the
work copy is restored byte for byte after every attempt.

Mutation kinds (counts): statement removal 6, raised exception 6, comparison
operator flip 4, constant off-by-one 3, argument swap 3, early return `None` 3,
arithmetic operator flip 3, wrong-value return 3, adjacent-statement swap 2
(a two-line edit), negation removal 2, boolean constant flip 2, boolean operator
flip 2, duplicated statement 1. Ten cases insert one line (net +1); two edit two
consecutive lines. Edits never change the line count by more than one, as the
evaluator's span check requires.

### Where the failure surfaces

The category is computed from the stack trace text alone (no model output):

| Category | Cases | Meaning |
| --- | ---: | --- |
| root cause is the innermost production frame | 19 | the changed function raised |
| root cause appears deeper in the stack | 3 | a function the changed function called, or the test, raised afterwards |
| root cause absent from the stack | 18 | the failure surfaces in the test (for example a failed assertion on a wrong value) |

So 21 of 40 cases surface away from the root cause. The "deeper in the stack"
group is small (3 of 40) because the pytest tracebacks of these projects are
short; it is a limitation, not a design choice.

### Structural pool check (no model, no ranking score)

The shipped retrieval proposes only functions that appear in in-repository stack
frames. Loading all 40 cases through `evaluation/product_pool.py` (the same
code path the Gate C driver uses) gives:

| Check | Result |
| --- | --- |
| cases loaded through the product pool, 4 repositories | 40 |
| empty retrieval pools (abstentions) | 0 |
| labelled root-cause function present in the shipped retrieval pool | 22 of 40 (0.55) |
| ... innermost-frame cases | 19 of 19 |
| ... deeper-in-stack cases | 3 of 3 |
| ... absent-from-stack cases | 0 of 18 |

The 18 absent-from-stack cases can never be found by the shipped retrieval, by
construction. That caps the product-pool metrics at 0.55 for every method that
only reranks the shipped pool, so product-pool scores must be read next to the
pool recall. The labelled-pool (`--pool supplied`) comparison does not have this
cap. Cases were not selected, filtered or edited based on pool recall or on any
model output.

## Evaluator change this set needs

Real repositories repeat method names (`__init__`, `setUp`, ...). The product
pool previously rejected the whole run if any retrieved function name was not
unique in its file. With real paths that happens for these repositories, so
`evaluation/product_pool.load_product_cases` and `evaluation/run_gate_c.py` gained an
opt-in `--allow-ambiguous-retrieved-names` flag. With it, only the exact
(name, start line, end line) retrieved unit must be unique. Labelled root-cause
functions must still have a name that is unique in their file (the generator
guarantees this). The flag is recorded in the report. The default is unchanged,
so earlier reports (whose pools were empty) are unaffected.

## Protocol

1. **One use per selected candidate.** Choose the candidate on the dev set. Run
   Gate C v2 once for that candidate and the shipped baseline (three fixed seeds
   inside the run). Do not tune, retrain, or reselect on these results; if a
   second look is ever needed it needs a new set.
2. **Disjointness.** `train_manifest.json` lists every repository used for
   training or selection (the v0.2 train and dev repositories and the legacy
   training repositories). The evaluator rejects any overlap. When a new training
   run adds repositories, list them in a JSON file and regenerate only that
   manifest:

   ```bash
   python -m training.corpus.gatec_v2 --root ~/.local/share/failurelens/gatec-v2 \
     --python training/corpus/sandbox/sandbox_python.sh manifest \
     --train-repos-file extra_training_repos.json
   ```

   The command fails if a listed repository is one of the four evaluation
   repositories or the first hold-out. The records and trees do not change, so
   their digest is stable; the train-manifest digest changes and is part of the
   run's metadata.
3. **Run** (CPU, offline, evaluator environment) with the existing driver:

   ```bash
   python evaluation/run_gate_c.py --records $R/records.json --records-sha256 <digest> \
     --source-manifest $R/source_manifest.json --train-manifest $R/train_manifest.json \
     --tree-manifest $R/tree_manifest.json --pool product --allow-ambiguous-retrieved-names \
     --model-artifact ... --model-sha256 ... --model-label ...
   ```

   Run `--pool supplied` as the labelled-pool companion. Thresholds are the frozen
   Gate B values in `docs/decision-gate.md`; the report compares them
   mechanically and makes no decision.
4. **Aggregate only.** Reports contain counts, metrics, intervals and digests. Never
   print or commit cases, labels, source text, logs or per-case outputs.

## Limitations

- 40 synthetic cases from 4 small, well-tested libraries; one mutation per case.
  Results do not generalize to arbitrary real-world failures.
- Records are produced by injecting an edit that the project's own tests detect,
  so they lean toward well-tested code.
- The pool recall cap above (0.55) applies to the shipped retrieval, and a model
  that only reranks that pool inherits it.
- Only 3 cases have the root cause deeper in the stack.
- `hukkin/tomli` and `jazzband/docopt-ng` were tried and dropped before
  generation of the final set because their test suites gave almost no detected
  edits or do not run under the sandbox's pytest; this was an infrastructure
  decision made before any model was run.
