# Issue #17 — private synthetic training handoff

Status: **v2 private correction completed** on 2026-10-30; v1 below is retained as historical evidence and is **not label-approved**. This is not a public dataset, a trained model, or evidence of production failure-localization accuracy. The individual records and generators for both versions remain outside Git in the private locations below (directory mode 700; private artifacts mode 600). This same-user permission boundary is procedural, not isolation from processes using that account.

## Rights and source reservation

| Train-only source | Pin | Rights evidence and decision | Notice obligations |
| --- | --- | --- | --- |
| `pallets/click` | `2247b35ea1c47c727d7a06e51fa280e12a863ff6` | [BSD-3-Clause LICENSE.txt](https://github.com/pallets/click/blob/2247b35ea1c47c727d7a06e51fa280e12a863ff6/LICENSE.txt); **Usable** for local source-based synthetic training and self-generated probe logs. | Retain copyright, conditions, disclaimer for any source/derived redistribution; no endorsement. The selected `src/click/utils.py` is project code. `src/click/parser.py` carries optparse/PSF attribution and was excluded. |
| `python-attrs/attrs` | `c8bb9bfaa2843e7844b5bb13bed4370b51feec4f` | [MIT LICENSE](https://github.com/python-attrs/attrs/blob/c8bb9bfaa2843e7844b5bb13bed4370b51feec4f/LICENSE); **Usable** for local source-based synthetic training and self-generated probe logs. Selected `src/attr/{converters,filters,_funcs}.py` have MIT SPDX headers. | Include copyright and permission notice with any source/substantial-copy redistribution. |

Both licenses permit modification and use, including this local training purpose; neither upstream grants rights to unrelated user logs, which were not collected. No vendored, test-fixture, or third-party-attributed files were used. There is **no authorization here to release the derived records**. The public artifact is aggregate metadata only.

The private manifest reserves only `pallets/click` and `python-attrs/attrs` for training and explicitly excludes eval-only `psf/requests`, `pallets/werkzeug`, and `pypa/pip`, as named in [the published Gate B reservation](gate-b-independent-eval-handoff.md). The train/eval repository intersection is empty. No held-out records, labels, or generator were opened; case separation is inferred from disjoint source repositories, not from inspecting eval labels.

## Reproducibility and validation

- Version: `failurelens-train-issue17-v1` / private generator `issue17-private-1.0`; Python 3.12.3; source imports from each pinned checkout's `src/`; no network in probes.
- Deterministic method: one early-return mutation of a selected source function per case, isolated clean and mutated subprocesses, then restore original bytes. Test command template: `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -c '<private offline assertion probe>'` from the relevant source checkout. Exact commands, mutations, exits and logs reside in the private records/generator; they are deliberately not published.
- Attempted and accepted: **12/12**; `pallets/click` **6**, `python-attrs/attrs` **6**. Clean exit 0: **12/12**; mutated exit 1 with assertion/type failure: **12/12**. No install/network/environment failures accepted.
- `datasets/schemas/record.schema.json` (Draft 2020-12 + format checker): **0 violations**. `failurelens.quality.validate_record_set(records, redistributed=True)`: **12/12 passed, 0 rejections**. Duplicate IDs/signatures: **0**. Secret/PII screener hits in accepted logs: **0**. These heuristic gates do not prove the absence of all sensitive data or semantic label errors.
- Private `records.json` SHA-256: `ce9112ab48f4edca14758d2b64e16b605e878c042c65f26af0c8aa5675185b1c`. Recheck privately with `sha256sum ~/.local/share/failurelens/train-issue17-v1/records.json`; rerun with `python3 ~/.local/share/failurelens/train-issue17-v1/generate.py` (regeneration changes timestamps and the digest).

The v1 candidate and rationale summary above is historical: independent review identified non-source negative locations and explanations that incorrectly claimed assertion failure when an exception occurred earlier. Do not use v1 labels as approved training truth. These are simple synthetic early-return faults, heavily concentrated in a few modules, not natural defects. No GPU training, model upload, or external dataset publication occurred.

## Private v2 correction (2026-10-30)

The corrected private version is at `~/.local/share/failurelens/train-issue17-v2/`; v1 remains unchanged. The private generator and individual records are **not** in Git. Private `records.json` SHA-256: `74909f51b588c311de798bee29b748cfc1ead65fd2948053c8f3a1c729e465c1`.

- **12/12 accepted**, split **6/6** between the same two pinned, rights-reviewed train-only source repositories. All 12 clean probes exited 0; all 12 changed probes exited 1. There were 4 early-return, 4 wrong-value-return, and 4 explicit-exception mutations; observed failures comprised 4 assertion, 3 type, 1 attribute, and 4 runtime exceptions.
- Each negative is now an existing same-module source function, confirmed at its pinned source lines and observed uncalled in both clean and changed profiled probes; the positive function was observed called in both. Rationales distinguish assertions reached after a changed return value from exceptions that occurred before the assertion. Source locations, terminal exceptions, and rationales were reviewed for all 12.
- Draft 2020-12 schema: **0 violations**. Semantic quality gate: **12/12 passed, 0 rejections**. Duplicate IDs and exact signatures: **0**. Pairwise near-signature check (source repository, location, mutation method, probe; similarity threshold 0.90): **0 flagged**, maximum 0.832. Secret/PII heuristic gates: **0 hits**. Source clones were restored clean after each mutation.
- Train/eval source-repository intersection remains **0** by reservation. No held-out records, labels, or generator were opened. This is not a case-level or semantic proof of non-leakage; it relies on disjoint pinned source repositories.
- Limitations: only 12 synthetic, single-function, single-probe examples across a small number of modules and exception types. Negatives are uncalled adjacent functions rather than hard executed-but-noncausal candidates. The three insertion-style mutation mechanisms are not representative of natural production defects. The automated checks and source review do not constitute independent human label approval, Gate C accuracy, redistribution authorization, or model-readiness.

No model training, Hugging Face publication, or case-level GitHub release occurred.
