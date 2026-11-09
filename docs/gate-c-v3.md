# Gate C v3: a second fresh final-evaluation set

**Status: built and frozen on 2026-11-09. No model and no baseline has been run
on it and there are no accuracy results.** This page describes the set in
aggregate only. Records, logs, source snapshots, failing trees, labels and
per-case metadata are private, live outside Git, and must never be committed or
quoted. It is a measurement set, not a Gate C decision, and it authorizes
nothing: training, publication and adoption need separate approval.

## Why a third set

Gate C v2 is **consumed**: its aggregate numbers have been looked at for the
v0.4 final measurement (see `docs/releases/v0.4/REPORT.md`). A confirmation of
the v0.5 call-graph baseline, or of any later candidate, therefore needs a set
nobody has looked at. Gate C v3 is that set. A second look at v3 would again need
a new set.

## What it contains (aggregate)

| Item | Value |
| --- | --- |
| Cases | 60, all function-labelled |
| Repositories | 5 (`PyCQA/pyflakes`, `Python-Markdown/markdown`, `python-hyper/h11`, `python-hyper/hpack`, `pallets/markupsafe`) |
| Cases per repository | 14, 14, 14, 14, 4 (markupsafe is small: only 6 candidate edits were accepted) |
| Licenses | MIT (pyflakes, h11, hpack) and BSD-3-Clause (markdown, markupsafe); pins and license-blob digests in `docs/gate-b-rights-ledger.md` |
| Records SHA-256 | `de9506f67cd38b464c029d54bc91a5ddfc6d0d89b6dbd4433e0857d3018333cc` |
| Source manifest SHA-256 | `41dcd5f78f9ca37c705636d3c424abead7a13eeb3f8ea9ca3ff70c1cddd7f314` |
| Tree manifest SHA-256 | `5800535eb823cd5bd6241896f8fe1232e852a88228f3388b2c341bd1dfcf86df` |
| Train manifest SHA-256 | `58e812c19609a6b58cd6400117a95696f23b2dcdd8110697102899d8d1071c83` (regenerated per training list, see below) |
| Generator | `training/corpus/gatec_v3.py` (`gatec-v3-1.0`), seed `20261109`, built on `training/corpus/gatec_v2.py` |
| Third-party code execution | only inside the network-less Docker sandbox of `training/corpus/sandbox/` |

Every case passes the schema and the semantic gates, the clean tree passes the
failing test, the edited tree fails it (exit code 1, native traceback), and the
work copy is restored byte for byte after every attempt. The set is disjoint from
every repository used before: the v0.2/v0.3 train and dev repositories, the
legacy training repositories, the first hold-out and the Gate C v2 sources.
`gatec_v3.validate_v3_table` fails closed on any overlap.

### Where the failure surfaces (the design target for a call-graph test)

The category is computed from the stack trace text plus an **independent**
name-based static call check written for the builder (not the library's call
graph), so the set is not designed around the baseline under test. A relation
counts only through a name that is defined exactly once in the repository, so a
common name such as `get` cannot create a spurious relation.

| Category | Cases | Meaning |
| --- | ---: | --- |
| root cause is the innermost production frame | 14 | the changed function raised |
| root cause appears deeper in the stack | 12 | the failure surfaces below it, or in a test frame after it |
| absent from the stack but call-related | 18 | not in the stack, but directly calls or is directly called by a production stack frame |
| absent from the stack and not call-related | 16 | not in the stack and no direct call relation (an honest negative control) |

The target shares were 20% / 20% / 35% / 25%. The achieved mix is
23% / 20% / 30% / 27%. To reach it, the per-repository generation was raised
(oversupply of up to 80 execution-validated candidates for the three larger
repositories) before selection; the candidate pool before selection had 85 / 14 /
19 / 105 cases per category, so the two middle categories were taken almost
exhaustively. Selection is stratified on repository, category and mutation kind
with the fixed seed and uses structural metadata only.

"Callee change surfacing elsewhere" is represented by the two absent-from-stack
categories (the changed function is not on the stack while the failure still
surfaces in the project's own code). It is not a separate mutation kind here.

### Edits

18 mutation kinds appear, among them wrong-value return (9), early-return `None`
(5), swapped call arguments (5), statement removal (4), raised exception (4),
multi-line edit (4), call-argument swap (3), dropped call argument (3),
duplicated statement (3), constant off-by-one (3), negation removal (3),
comparison-operator flip (3), neutralised guard clause (2), adjacent-statement
swap (2, a two-line edit), arithmetic-operator flip (2), boolean-operator flip
(2), boolean constant flip (2) and loop or slice bound off-by-one (1). 8 cases are
multi-line edits and 12 insert one line (net +1). Edits never change the line
count by more than one, as the evaluator's span check requires.

### Structural pool check (after the freeze; no model, no ranking score)

Run only after the set was frozen. Loading all 60 cases through
`evaluation/product_pool.py` (the code path the Gate C driver uses, with the
shipped lexical retrieval pinned as the comparator) gives:

| Check | Result |
| --- | --- |
| cases loaded, 5 repositories | 60 |
| empty retrieval pools (abstentions) | 3 (all absent-and-unrelated cases) |
| labelled root-cause function present in the shipped retrieval pool | 26 of 60 (0.43) |
| ... innermost-frame cases | 14 of 14 |
| ... deeper-in-stack cases | 12 of 12 |
| ... absent-from-stack, call-related cases | 0 of 18 |
| ... absent-from-stack, unrelated cases | 0 of 16 |

The shipped lexical retrieval proposes only functions that appear in
in-repository stack frames, so the 34 absent-from-stack cases can never be found
by it. That caps product-pool metrics at 0.43 for any method that only reranks
that pool. A whole-repository pool (as in `evaluation/run_full_pool.py` and the v0.5
driver) does not have this cap. Cases were not selected, filtered or edited
based on pool recall or on any model or baseline output.

## Evaluator flag this set needs

As for v2, real repositories repeat method names, so the product pool needs
`--allow-ambiguous-retrieved-names` (an opt-in flag of `evaluation/run_gate_c.py`;
see `docs/gate-c-v2.md`). The default stays strict.

## Protocol

1. **One use per selected configuration.** Choose the configuration on a dev set
   first. Run Gate C v3 once for it and for each comparator (the shipped
   baseline, the lexical comparator), with the fixed seeds inside the run. Do not
   tune, retrain or reselect on these results; a second look needs a new set.
2. **Disjointness.** `train_manifest.json` lists every repository used for
   training or selection. When a later training run adds repositories, list them
   in a JSON file and regenerate only that manifest (the records and trees do
   not change):

   ```bash
   python -m training.corpus.gatec_v3 --root ~/.local/share/failurelens/gatec-v3 \
     --python training/corpus/sandbox/sandbox_python.sh manifest \
     --train-repos-file extra_training_repos.json
   ```

   The command fails if a listed repository is one of the evaluation repositories
   or the first hold-out.
3. **Aggregate only.** Reports contain counts, metrics, intervals and digests.
   Never print or commit cases, labels, source text, logs or per-case outputs.
4. **Thresholds.** The frozen Gate B values in `docs/decision-gate.md` apply
   unchanged; comparisons are mechanical and make no decision.

## Limitations

- 60 synthetic cases from 5 small, well-tested libraries; one mutation per case.
  Results do not generalize to arbitrary real-world failures.
- The 5 repositories are uneven: four contribute 14 cases and `markupsafe` only 4.
- Records are produced by injecting an edit that the project's own tests detect,
  so they lean toward well-tested code.
- The category is a structural proxy (stack text plus a name-based call check),
  not a human judgement of causality.
- 8 of the 60 cases are multi-line edits.
- Repositories tried and dropped before generating the final set, on
  infrastructure grounds and before any model or baseline output existed:
  `python-semver/python-semver` (tests import helper modules from `docs/`),
  `scrapy/w3lib` (needs a package the sandbox image lacks), `python-hyper/hyperframe`
  (failure logs without a usable traceback), `tobgu/pyrsistent` (its suite does
  not run under the sandbox's pytest) and `tartley/colorama` (1 accepted edit).
