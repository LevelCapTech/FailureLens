# Model card — FailureLens

> **Status: trained artifact exists; Gate C evaluation pending; no publication
> decision yet.** A fine-tuned bi-encoder (`failurelens-coderankeembed-stage2-v3`)
> has been trained and its artifact verified. Gate C (independent held-out
> evaluation) has not yet run. Gate D (adoption/publication) requires an explicit
> human-owner decision after Gate C. Do not cite any quality numbers from this
> card that are not produced by the evaluation harness on committed data.

## Summary

| Field | Value |
|-------|-------|
| Name | FailureLens CodeRankEmbed Stage 2 v3 |
| Task | Fault localization — rank candidate code locations for a failure |
| Base model | `nomic-ai/CodeRankEmbed` @ `3c4b60807d71f79b43f3c4363786d9493691f8b1` (MIT) |
| Fine-tuning method | Last NomicBertBlock encoder layer only (LoRA-equivalent partial fine-tune) |
| Trainable params | 9,440,256 / 136,731,648 total (~6.9%) |
| Artifact SHA-256 | `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4` |
| Artifact path (private) | `~/.local/share/failurelens/experiment-issue19-stage2-v3/model_final.safetensors` |
| Format | safetensors only (`.bin` rejected at load time) |
| Training hardware | NVIDIA GeForce RTX 5060 Ti (16 GB) |
| Runtime | CPU or CUDA; offline after model cache |
| Code git SHA | `795b0a1c6f94198d93ebd36e45d3d61e8574549e` |
| Trained weights in Git | No |
| Trained weights on HF Hub | No (Gate D not approved) |
| Gate C evaluation | Not yet run |
| Gate D adoption/publication | Not yet approved |
| Shipped baseline (for comparison) | `analysis_method = baseline-stacktrace-lexical` |

## Intended use

- Help developers, CI systems, and AI coding agents **narrow down** where a
  failure likely originates, with auditable evidence.
- Input: an error log / stack trace + a source repository. Output: ranked
  candidate locations (file / function / line span) with evidence.

### Out of scope

- Not an LLM API wrapper, generic code search, or error-message summariser.
- Does **not** guarantee a fix is correct — it suggests candidates.
- Not a multi-language tool yet (Python traceback support first).

## Architecture

Fine-tuned bi-encoder over `nomic-ai/CodeRankEmbed` (NomicBERT, 137M params,
8192-token context). Only the last encoder layer is trainable. Trained on
InfoNCE loss with positive/negative code-function pairs reconstructed from
private synthetic failure records (see `docs/training.md`).

At inference, the v3 bridge (`evaluation/v3_bridge.py`) applies the exact
Stage 2 query transform (sanitized `error_log.strip()`); the model scores
each candidate code fragment against the query in a shared retrieval pool.

## Training data

- Dataset: `failurelens-train-issue17-v2`
  (SHA-256: `74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1`)
- 12 synthetic failure records, 2 rights-cleared repositories:
  `pallets/click` (train, 6 records) and `python-attrs/attrs` (val, 6 records)
- Repo-level split to avoid cross-split leakage
- Positive text: mutated source function (actual failing code), sanitized of
  all generator markers, private paths, and injected exception messages
- Negative text: clean source function confirmed absent from the failure trace

## Training configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR scheduler | CosineAnnealingLR |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| InfoNCE temperature | 0.07 |
| Batch size | 2 |
| Epochs | 3 |
| Seed | 42 |
| Max sequence length | 512 |

## Training results (v3, seed=42)

| Epoch | Train loss | Val loss |
|-------|-----------|---------|
| 1 | 0.7298 | 0.7346 |
| 2 | 0.4976 | 0.7241 |
| 3 | 0.5162 | 0.7203 |

Elapsed: 3.5 s. Peak GPU: 885 MiB / 15,847 MiB. Checkpoint/resume verified
(epoch 3 loss bit-exact between full run and resume from epoch-2 checkpoint).

## Limitations and caveats

- **Tiny synthetic dataset**: 6 training / 6 validation examples from one
  repository each. Metrics above reflect fitting this synthetic set; no
  claim about generalisation to real-world repositories.
- **No independent quality result yet**: Gate C (held-out evaluation on ≥30
  cases across ≥3 rights-cleared repositories) has not run. The frozen Gate B
  thresholds (function Top-1 ≥ 0.4, MRR ≥ 0.6) remain unverified.
- **Comparator limitation**: the `supplied-pool-stack-lexical` baseline adapter
  is not the shipped `baseline-stacktrace-lexical` analyzer; a valid
  shipped-baseline Gate C delta requires the common-pool protocol from
  `docs/shipped-baseline-common-pool.md`.
- Only Python tracebacks parsed today; other formats return no candidates.
- Lexical matching is shallow; the embedding reranker adds semantic matching
  but has not been independently evaluated.

## Baseline (for comparison while waiting for Gate C)

The shipped deterministic baseline (`analysis_method = baseline-stacktrace-lexical`):
`score = 0.6·depth + 0.2·exception_site + 0.4·lexical`

Reproduced with:

```bash
python evaluation/run_baseline_eval.py
```

Gate A measured results (n=1, single fixture case, not generalisable):
- file: Top-1/5/10 = 1.0, MRR = 1.0
- function: Top-1 = 0.0, Top-5/10 = 1.0, MRR = 0.5

## Gate status

| Gate | Status |
|------|--------|
| Gate A — Pilot + baseline measurement | Done (2026-10-27) |
| Gate B — Acceptance criteria frozen | Done (2026-10-29, human-owner approved) |
| Implementation Go | Done (2026-10-29, human-owner approved) |
| Gate C — Independent held-out evaluation | **Not yet run** (blocked by comparator protocol review) |
| Gate D — Adoption / publication | **Not yet approved** |

## Licensing

- Repository source: Apache-2.0
- Base model (`nomic-ai/CodeRankEmbed`): MIT
- Fine-tuned weights: **license not yet determined**. A derivative of an MIT
  base model is compatible with MIT, but the license for the published weights
  must be decided and verified by the human owner before any release. Apache-2.0
  is not assumed to apply to weights.
- Training data: private synthetic records; rights reviewed per `docs/data-contract.md`.

## Provenance

- Code: `LevelCapTech/FailureLens` at `795b0a1c6f94198d93ebd36e45d3d61e8574549e`
- Training: `training/finetune.py` (InfoNCE, safetensors, `weights_only=True`)
- Sanitization audit: PR #25 issuecomment-6513753530 (independent steward)
- Artifact: `experiment-issue19-stage2-v3/model_final.safetensors`
  (SHA-256: `506492b034fadb3d7ee6e7288d04b66dfa477c25b1de41d9d70166353cd343d4`)
- Tainted artifacts (do not use for Gate C): v1 (`7520d7aa...`), v2 (`e91955b4...`)
