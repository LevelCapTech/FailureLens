# Gate B independent evaluation set: local handoff (Issue #11)

On 2026-10-29 JST the data steward generated **30 synthetic pass-before/fail-after cases** on the configured `rtx5060ti` node: 10 each from `psf/requests`, `pallets/werkzeug`, and `pypa/pip`. The pinned commits and rights determinations are in [gate-b-rights-ledger.md](gate-b-rights-ledger.md). These are single-line behavioral mutations exercised by deterministic, offline assertion probes. They are a feasibility set, not a claim of trained-model accuracy or realistic production failure coverage.

The held-out artifact is kept off the public repository at `~/.local/share/failurelens/heldout-gateb-v1/`. It contains `generate.py` (exact mutations and probes), `records.json` (case labels, logs, execution results and per-case provenance), and `split.json` (repository-level split metadata). `records.json` SHA-256: `823723ce0130bc489285b6a39491d62c3637bff3712719bac7a2eb3ea2d937f4`. Do not expose the case labels or generator to model development; independent evaluation must preserve this holdout.

Validation observed on this node:

| Check | Result |
|---|---:|
| Clean-state probe passed | 30/30 |
| Mutated-state probe failed for the intended behavioral change | 30/30 |
| JSON Schema violations | 0/30 |
| `failurelens.quality.validate_record_set(..., redistributed=True)` accepted | 30/30 |
| Distinct pinned source repositories | 3 |
| Duplicate record IDs or near-duplicate signatures | 0 |
| Train/eval repository overlap | 0 |

The manifest reserves these three repositories for **eval only**. It records zero actual training records because no trained FailureLens model or production training dataset exists; the in-repo sample fixture is a development-only source outside the eval repositories. Thus the current repository reservation has zero overlap, but a non-leaking split against a future training corpus cannot yet be verified. This is not evidence of a trained model or an independent evaluator's report. The probes are synthetic and small; label and representativeness review is still required before Gate C use.

No dataset contents are included here: the data contract and rights ledger do not authorize a public derived-dataset release, and a public PR would reveal held-out labels. The files are mode 600 in a mode 700 directory, but agents running under the same Unix account can still read them; this is procedural separation, not enforced isolation. The coordinator/independent evaluator must arrange an access-controlled handoff before Gate C without publishing labels to model development. Any eventual publication is a separate owner decision, with upstream notices and a renewed rights/privacy review.

## コードレビューフィードバック対応

- Review thread: https://github.com/LevelCapTech/FailureLens/pull/12#pullrequestreview-5697485147
  - Comment: `docs/gate-b-independent-eval-handoff.md:5`
    - Request type: change request
    - 対応内容: 絶対パス `/home/<user>/...` をホーム相対パス `~/.local/share/failurelens/heldout-gateb-v1/` に置換し、個人識別子を除去。
