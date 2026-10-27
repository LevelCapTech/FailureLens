# Design gate — Go/No-Go criteria and handoff

This is the **hub** for the AI-model design gate (issue #3). It sequences the
gates, frames the Go/No-Go criteria, and records ownership/handoff. It builds on
the baseline delivered at `main` commit `611bc0ea9aed8118b7afd637f97591f344eda072`.

> **What this is.** A **design gate**: an agreement on the data contract,
> evaluation protocol, and decision criteria *before* large implementation.
>
> **What this is not.** It is **not** authorization for full-scale training, data
> collection at scale, or any public dataset/model release. Those remain separate,
> human-approved decisions.

## Design-gate documents

| Deliverable | Document |
|-------------|----------|
| Requirements specification | `docs/requirements.md` |
| Data sources/rights matrix, pilot, quality gates | `docs/data-contract.md` |
| Independent evaluation protocol | `docs/evaluation-protocol.md` |
| Go/No-Go sequencing and handoff | this document |

## Sequencing (ordered gates)

The gates run **in order**; do not begin a later stage before the earlier gate is
cleared.

1. **Gate A — Pilot feasibility & baseline measurement.** Establish that a
   **rights-cleared** data pilot is feasible (`docs/data-contract.md`) **and**
   measure the existing baselines on identical conditions
   (`docs/evaluation-protocol.md`). Output: feasibility finding + measured
   baseline numbers.
2. **Gate B — Freeze the acceptance criteria.** *After* Gate A, propose and obtain
   **human-owner approval** for the acceptance thresholds, resource budgets,
   sample size, and uncertainty treatment — **before** any trained-model result on
   held-out data is inspected. Output: the frozen table below.
3. **Gate C — Trained-model evaluation (separate).** Only after Gate B is frozen,
   the independent evaluator runs the comparison and produces a report. Execution
   is a **separate** step from this design gate.
4. **Gate D — Adoption / publication (separate).** The human owner decides on
   adoption and any dataset/model publication, subject to rights and licenses
   (`docs/security.md`, `docs/model-card.md`). This is **separate** from Gate C.

## Acceptance-criteria freeze (values set at Gate B)

These values are deliberately **unset**. They must be proposed **after** pilot and
baseline measurement (Gate A) and frozen with **human-owner approval** (Gate B),
**before** inspecting any trained-model held-out results. **Do not invent them.**

| Criterion | Value | Set when | Approver |
|-----------|-------|----------|----------|
| Primary KPI — function-level Top-1 | **TBD** | Gate B (post-pilot) | Human owner |
| Primary KPI — function-level MRR | **TBD** | Gate B (post-pilot) | Human owner |
| Minimum improvement over best baseline | **TBD** | Gate B (post-pilot) | Human owner |
| Evaluation case count (sample size) | **TBD** | Gate B (post-pilot) | Human owner |
| Evaluation repository count | **TBD** | Gate B (post-pilot) | Human owner |
| Uncertainty treatment | **TBD** | Gate B (post-pilot) | Human owner |
| Latency budget | **TBD** | Gate B (post-pilot) | Human owner |
| Memory budget | **TBD** | Gate B (post-pilot) | Human owner |
| Model-size budget | **TBD** | Gate B (post-pilot) | Human owner |

Function-level Top-1 and MRR are the primary KPIs **where function labels exist**;
file-level metrics and resource measurements are reported separately and never
substituted for function labels (`docs/evaluation-protocol.md`).

## Ownership and handoff

| Role | Owner | Responsibility |
|------|-------|----------------|
| Coordinator | `failurelens-coordinator` | Issue owner/review; **recommends** Go/No-Go; opens separate `[IMPLEMENT]` issues. |
| Data | `failurelens-data-steward` | Rights matrix, pilot, label review, non-leaking splits. |
| Model feasibility | `failurelens-model-developer` | Base-model/training feasibility (no scale-up here). |
| Independent evaluation | `failurelens-independent-evaluator` | Candidate parity, metrics, reproducible reports. |
| Human owner | Project maintainer | **Approves** the design gate and later adoption/publication. |

- The coordinator **recommends** Go/No-Go; the **human owner approves**.
- Large cross-cutting code changes belong in separately scoped `[IMPLEMENT]`
  issues and may be assigned to GitHub Copilot by the coordinator.
- Specialists use the configured **rtx5060ti** node without sandboxing and their
  own Discord channels. Do not begin later stages before their gates clear.

## Separation of duties

**Design (this gate)**, **evaluation execution** (Gate C), and the **final
adoption decision** (Gate D) are distinct. Approving this design gate does not
approve evaluation outcomes or adoption, and no public dataset/model release or
full-scale training is authorized by issue #3 alone.

## Deliverable checklist

- [ ] Requirements separate existing vs planned features, with users, Python-first
      scope, inputs/outputs, exclusions, risks, and owners (`docs/requirements.md`).
- [ ] Data source/rights matrix and pilot plan cover usable vs pending sources,
      label review, semantic quality gates, secret/PII screening, versioning, and
      non-leaking splits, using no unapproved data (`docs/data-contract.md`).
- [ ] Independent evaluation protocol fixes candidate parity, metrics,
      missing-label handling, comparison baselines, reproducible reporting, and
      failure/bias analysis (`docs/evaluation-protocol.md`).
- [ ] Numeric acceptance thresholds, resource budgets, sample size, and
      uncertainty plan are **frozen at Gate B with human approval**, before final
      trained-model evaluation (table above; no invented values).
- [ ] Coordinator obtains **human Go/No-Go** before separate implementation
      issues; evaluation execution and final adoption remain separate.
