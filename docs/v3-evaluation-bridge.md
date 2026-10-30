# Opt-in v3 evaluation bridge (Issue #26)

`evaluation/v3_bridge.py` is evaluator-owned, import-safe code. It does not load
private inputs, weights, or a model on import. The checkpoint must be supplied
at runtime as `FAILURELENS_V3_ARTIFACT`; `FAILURELENS_V3_CACHE` is a required
local audited base-model cache; `FAILURELENS_V3_DEVICE` is `cpu` (default) or
`cuda`. Use an isolated evaluator environment with the `ml` and `train` extras.
Only the exact v3 safetensors artifact SHA-256 and the recorded Stage 2 train
digest and audited CodeRankEmbed revision are accepted. Metadata is verified
before model load; no optimizer pickle, v1/v2 artifact, remote weight publish,
or retraining is part of this bridge. `model_factory(seed)` is the direct
`independent_runner` factory; `model_spec(Path(...))` supplies checked method
metadata. Batch size one bounds activation memory, while the runner measures
actual process RSS and latency. No performance result is implied. A local
base-model snapshot at the exact audited revision is mandatory; a cache with
unsafe `.bin` weights is rejected. Run with `HF_HUB_OFFLINE=1` to prevent Hub
requests. For a pre-measurement provenance check in an isolated evaluator venv:

```bash
HF_HUB_OFFLINE=1 FAILURELENS_V3_ARTIFACT="$ARTIFACT" \
  python -c 'import os; from pathlib import Path; from evaluation.v3_bridge import model_spec; model_spec(Path(os.environ["FAILURELENS_V3_ARTIFACT"]))'
```

The private runner may be invoked only after the comparator protocol is approved;
pass `evaluation.v3_bridge.model_factory` and its `model_spec` to
`run_independent` with the separately validated `load_private_cases` output.
Do not print case objects or the returned report to stdout.

The canonical request query is exact sorted compact JSON with only `error_log`
and `stack_trace`. The model sees only `error_log.strip()`, after verifying it
already equals the Stage 2 sanitization transform. This is the exact training
query feature; JSON keys and stack trace are never model text. A record needing
extra redaction or a different transform is incompatible and must fail, not be
silently rewritten. Candidate source is extracted from SHA-verified failing
snapshots by AST function name and both location bounds. A single inserted
mutation line can extend its containing function (`start+0,end+1`) or shift a
later function (`start+1,end+1`); any other range or ambiguous match fails.
The source-manifest provenance of the failing snapshot remains an evaluator
obligation as described in `private-evaluation-adapter.md`.

## Comparator limitation / Gate C hold

The runner's `supplied-pool-stack-lexical` method is *not* the product-shipped
`baseline-stacktrace-lexical` analyzer. The shipped analyzer first parses the
trace, resolves frames against a repository index, and retrieves only frame-
associated candidates. It returns `INSUFFICIENT_EVIDENCE` for no frames and
`NO_CANDIDATES` for unresolved frames. It cannot emit a full permutation of an
arbitrary supplied candidate pool, and its depth/exception-site ranking inputs
are not present in `RankRequest` (which has only source slices and JSON log/
trace). Replacing it with the supplied-pool heuristic changes the method.
Thus no comparable shipped-baseline Gate C delta exists from the current API.
A separately reviewed protocol redesign would need a common retrieval pool and
frame evidence, or explicitly change the comparator/KPI definition. Until then,
**do not run the held-out measurement or claim the frozen Gate C KPI**. Public
synthetic tests and `make check` are safe pre-acceptance verification only.
