# v0.3 private fault corpus (harder examples)

> **Status: built and digest-pinned (2026-11-07). Aggregate counts only.** It does
> not authorize training, release or the Gate C run, and no model result is implied.
> Records, clones, failing trees and manifests stay in a private directory outside
> Git. Do not cite quality numbers from this page.

The v0.2 candidates fit the dev set better (validation loss fell) but their ranking
scores barely moved: single-line flips are often solvable by lexical matching. The
v0.3 corpus keeps the v0.2 pipeline (`docs/corpus-v02.md`: pass-before/fail-after,
one exact edit, sanitised logs, schema and semantic gates, sandboxed execution) and
adds harder examples and four more train repositories.

## What is new

`training/corpus/mutations.py` enumerates, with `enumerate_mutations(..., hard=True)`:

| Hard kind | Shape |
| --- | --- |
| multi-line edit | two independent edits on nearby lines, one multi-line recipe |
| swapped call arguments | first two positional arguments of a call exchanged |
| dropped call argument | last positional argument removed |
| loop or slice bound off-by-one | `range(n)` -> `range(n - 1)`, `x[:n]` -> `x[:n - 1]`, lower bound `+ 1` |
| changed default value | integer/boolean/non-empty-string default in the `def` header |
| neutralised guard clause | `if cond: raise/return/continue/break` -> `if cond: pass` (two-line recipe) |
| callee change surfacing elsewhere | assigned by the builder: the changed function does not appear in the traceback while the traceback still has a frame inside the project |

Recipes are runs of complete lines; `old` occurs once in the function and the net line
change is 0 or +1, so `evaluation.private_adapter` (span shift) and
`evaluation.product_pool.verify_failing_tree` accept them unchanged. `apply_mutation`
handles several-line recipes. Every record also stores `root_in_trace` (whether the changed
function name appears in the traceback) in the private `mutations.json`.

The builder selects the corpus version with `--profile v03` (ids `failurelens-*-v03`,
seed `20261108`, three kinds per function, hard kinds sampled about twice as often).
`--mutation-set hard` restricts generation to hard kinds for top-ups.

## Repositories

Splits are by repository; train and dev share none, and none is `psf/requests`,
`pallets/werkzeug` or `pypa/pip`. Four new train repositories (names a-m,
permissive, pinned, license text verified; see `docs/gate-b-rights-ledger.md`):
`jawah/charset_normalizer`, `micheles/decorator`, `executablebooks/mdurl`,
`lepture/mistune`. The four v0.2 dev repositories are reused.

## Build (2026-11-07, sandboxed as in `docs/corpus-v02.md`)

```bash
docker build -t failurelens-corpus-sandbox:v02 training/corpus/sandbox   # unchanged image
ROOT=~/.local/share/failurelens/v03-corpus
SANDBOX=$PWD/training/corpus/sandbox/sandbox_python.sh
PYTHONPATH=packages/python/src:. python -m training.corpus.build prepare  --profile v03 --root $ROOT --python $SANDBOX
PYTHONPATH=packages/python/src:. python -m training.corpus.build generate --profile v03 --root $ROOT --python $SANDBOX \
  --train-target 55 --dev-target 32 --jobs 3
PYTHONPATH=packages/python/src:. python -m training.corpus.build finalize --profile v03 --root $ROOT --python $SANDBOX
```

Corpus digest (`corpus_sha256`):
`048301aea2e8eb6918b252416cea431b1c73505034d9f14952b2f4b69974417e`.
Overlap train/dev: 0 repositories; overlap with the Gate C hold-out: 0.

| Split | Kept | Input | Dropped by the gates | Hard-kind share | Root absent from trace |
| --- | ---: | ---: | --- | ---: | ---: |
| train | 351 | 409 | 48 duplicate, 8 PII-pattern, 3 environment failure | 51.9% (182) | 180 |
| dev | 98 | 110 | 8 duplicate, 3 environment failure, 1 PII-pattern | 58.2% (57) | 60 |

| Repository | Split | Records |
| --- | --- | ---: |
| pytoolz/toolz | train | 46 |
| jawah/charset_normalizer | train | 49 |
| lepture/mistune | train | 51 |
| kjd/idna | train | 44 |
| pallets/click | train | 42 |
| more-itertools/more-itertools | train | 35 |
| astanin/python-tabulate | train | 31 |
| jpvanhal/inflection | train | 20 |
| executablebooks/mdurl | train | 17 |
| pallets/itsdangerous | train | 16 |
| micheles/decorator | train | 0 |
| python-humanize/humanize | dev | 28 |
| mahmoud/boltons | dev | 29 |
| andialbrecht/sqlparse | dev | 27 |
| jmespath/jmespath.py | dev | 14 |

Kinds (train / dev): raised exception 46 / 9, callee change surfacing elsewhere 45 / 14,
swapped call arguments 40 / 10, dropped call argument 40 / 11, early-return None 22 / 3,
multi-line edit 22 / 10, neutralised guard clause 20 / 5, constant off-by-one 19 / 6,
comparison-operator flip 18 / 6, wrong-value return 16 / 2, boolean-operator flip 15 / 6,
arithmetic-operator flip 13 / 4, negation removal 12 / 3, loop or slice bound off-by-one
9 / 3, boolean constant flip 8 / 2, changed default value 6 / 4.

`training/configs/stage2_v03_{a,b,c,d}.json` carry the digest; the loader refuses a corpus
whose digest differs. `finetune_train.py` selects the v0.3 loader for
`"corpus": "failurelens-corpus-v03"`. Candidate `d` sets `trainable_encoder_layers: 2`.

## Limitations (stated up front)

- Synthetic edits of module-level functions only; natural defects differ. The hard kinds
  are harder for lexical matching, not a measure of real-world difficulty.
- `micheles/decorator` produced no record: its suite is a single file that the call tracer
  did not attribute to module-level functions, so the repository adds none.
- Mutations that no test detects are dropped (for example 107 of 162 attempts for
  `charset_normalizer`), so the corpus leans toward well-tested code.
- `load_product_cases` (shipped-baseline common pool) still rejects the dev set: the shipped
  retrieval returns functions whose names repeat in a file, which the verifier treats as
  ambiguous. Dev evaluation therefore uses the supplied-pool path, as for v0.2.
  `load_private_cases` accepts all 98 dev records.
- Labels follow from construction (one recipe) and passing tests; the automated gates are
  screeners, not independent human label review.
- `APPROVAL_ID` records the owner's delegation for the v0.3 cycle; it is a label, not an
  independent approval of sources or labels.
