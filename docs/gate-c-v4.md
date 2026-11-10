# Gate C v4: a third fresh final-evaluation set

**Status: built and frozen on 2026-11-10. No model and no baseline has been run
on it and there are no accuracy results.** This page describes the set in
aggregate only. Records, logs, source snapshots, failing trees, labels and
per-case metadata are private, live outside Git, and must never be committed or
quoted. It is a measurement set, not a Gate C decision, and it authorizes
nothing: training, publication and adoption need separate approval.

## Why a fourth set

Gate C v3 is **consumed**: the v0.5 independent confirmation looked at its
aggregate numbers (`docs/releases/v0.5/CONFIRMATION-REPORT.md`). Any further
confirmation, or the evaluation of a later candidate, needs a set nobody has looked
at. Gate C v4 is that set. It uses the generator, categories and seeded stratified
selection of Gate C v3 unchanged (`training/corpus/gatec_v4.py` reuses
`training/corpus/gatec_v3.py`); only the repository table differs.

## What it contains (aggregate)

| Item | Value |
| --- | --- |
| Cases | 60, all function-labelled |
| Repositories | 5 (`pyparsing/pyparsing`, `hukkin/tomli`, `r1chardj0n3s/parse`, `keleshev/schema`, `martinblech/xmltodict`) |
| Cases per repository | pyparsing 22, xmltodict 14, parse 12, schema 6, tomli 6 |
| Licenses | MIT for all five; pins and license-blob digests in `docs/gate-b-rights-ledger.md` |
| Records SHA-256 | `f221ed9adc94a177bed35e14912daaf561b9237486d85bee78da6ed881880a2b` |
| Source manifest SHA-256 | `2d5b5d3b2726a94ac62db928227941ef288c8e6b0e55f06252fc7137a000cbc3` |
| Tree manifest SHA-256 | `8016c8a839065a3e11495d55699a99426546f62136dbe199e4b84b0dbd128093` |
| Train manifest SHA-256 | `d4e3265f4a27655177ea1e74ece22349df6323149ebb7f47105ada761816fea5` (regenerated per training list) |
| Generator | `training/corpus/gatec_v4.py` (`gatec-v4-1.0`), seed `20261110`, built on `gatec_v3.py` / `gatec_v2.py` |
| Third-party code execution | only inside the network-less Docker sandbox of `training/corpus/sandbox/` |

Every case passes the schema and the semantic gates, the clean tree passes the
failing test, the edited tree fails it, and the work copy is restored byte for byte.
The set is disjoint from every repository used before, including the Gate C v3
sources and the repositories dropped for v3; `gatec_v4.validate_v4_table` fails
closed on any overlap.

### Where the failure surfaces

Categories are computed exactly as for v3 (stack text plus an independent
name-based static call check, not the library's call graph).

| Category | Cases |
| --- | ---: |
| root cause is the innermost production frame | 22 |
| root cause appears deeper in the stack | 6 |
| absent from the stack but call-related | 6 |
| absent from the stack and not call-related | 26 |

**This mix is far from v3's** (14 / 12 / 18 / 16) **and from the 20 / 20 / 35 / 25 target.**
The candidate pool before selection was 32 / 7 / 7 / 52 per category: the two small
categories could not reach their targets because only 7 candidates each existed
(the small libraries mostly use direct calls that the tests exercise as the
innermost frame, or very local helpers). Selection was not changed to
chase the mix, and it uses structural metadata only. A call-graph signal is
therefore tested here mainly on 12 stack-present and 6 call-related cases;
the 26 unrelated cases are a negative control.

### Edits

19 mutation kinds appear. 9 of the 60 cases are multi-line edits and 14 insert one
line (net +1). Edits never change the line count by more than one.

### Structural pool check (after the freeze; no model, no ranking score)

| Check | Result |
| --- | --- |
| cases loaded, 5 repositories | 60 |
| empty retrieval pools (abstentions) | 0 |
| labelled root-cause function present in the shipped retrieval pool | 28 of 60 (0.47) |
| ... innermost-frame / deeper-in-stack | 22 of 22 / 6 of 6 |
| ... absent-from-stack (call-related / unrelated) | 0 of 6 / 0 of 26 |

Product-pool metrics (rerank only the shipped retrieval pool) are capped at 0.47.
A whole-repository pool has no such cap.

## Protocol

Same as Gate C v3 (`docs/gate-c-v3.md`): one use per selected configuration;
choose the configuration on a dev set first; aggregate-only reports; the frozen
Gate B thresholds in `docs/decision-gate.md` apply unchanged; comparisons are
mechanical and make no decision. A second look needs a new set. When the training
list grows, regenerate only the train manifest:

```bash
python -m training.corpus.gatec_v4 --root ~/.local/share/failurelens/gatec-v4 \
  --python training/corpus/sandbox/sandbox_python.sh manifest \
  --train-repos-file extra_training_repos.json
```

## Limitations

- 60 synthetic cases from 5 small libraries; one mutation per case. Results do
  not generalize to arbitrary real-world failures.
- Uneven repositories (22 / 14 / 12 / 6 / 6); `pyparsing` contributes 37% of the cases.
- The category mix is dominated by innermost-frame and unrelated-absent cases
  (80% together); only 6 call-related cases.
- Records are produced by injecting an edit that the project's own tests detect,
  so they lean toward well-tested code. Of 400 attempted tomli edits only 10 were
  detected by its small suite.
- The category is a structural proxy, not a human judgement of causality.
- Repositories tried and dropped on infrastructure grounds, before any model or
  baseline output existed: `jquast/wcwidth` (suite needs a C extension),
  `Suor/funcy` (suite needs an absent helper package),
  `grantjenks/python-sortedcontainers` (no failure log with a usable traceback).
