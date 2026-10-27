# AGENTS.md — Single Source of Truth for FailureLens contributors (humans & AI agents)

This file is the **operational contract** for anyone — human or AI coding agent
(e.g. Claude Code, ChatGPT/Codex, and similar) — making changes in this
repository. Read it fully before editing. If something here conflicts with a
vaguer instruction elsewhere, **this file wins** unless a maintainer says
otherwise.

---

## 1. What FailureLens is (in one paragraph)

FailureLens analyzes **failure evidence** (error logs / stack traces) against a
**source repository** and returns **ranked candidate fault locations** with
structured, auditable **evidence**. It distinguishes *where an error surfaced*
from *the code likely responsible*. Today it ships a **working, dependency-light
baseline** (stack-trace + lexical). Model-based backends are **optional** and a
FailureLens-trained model **does not exist yet**.

## 2. Non-negotiable honesty rules (read twice)

These come directly from the project charter. Violating them is worse than
writing no code:

1. **Never fabricate** trained models, weights, benchmark numbers, or accuracy
   claims. No trained FailureLens model exists; do not imply one does.
2. **Measure, don't assert.** Any accuracy/benchmark figure must be produced by
   `evaluation/run_baseline_eval.py` (or a successor harness) on committed data.
3. **Separate working from planned.** Code that runs today vs. future design must
   be clearly labelled (see each directory README).
4. **No forced network / downloads.** The baseline must run offline on CPU with
   only core deps. Never make model downloads or external APIs mandatory.
5. **Don't invent locations.** If a file/line can't be determined, the schema
   field is `null` — never a guessed number.
6. **Scores are not probabilities** until calibrated; keep `score_type` honest
   (currently `heuristic`).
7. **Respect licenses.** Apache-2.0 covers *this source*. A model's license
   depends on its base model + training data. Some comparison models are
   non-commercial — never redistribute their weights here.
8. **No secrets, no personal data** in code, data, fixtures, or logs.

## 3. Repository map

```
packages/python/              # The installable library + CLI (the product)
  src/failurelens/
    __init__.py               # Public exports
    api.py                    # FailureLens facade (from_baseline/from_pretrained/diagnose/index)
    cli.py                    # `failurelens` Typer CLI
    config.py                 # IndexConfig, DiagnoseConfig (Pydantic)
    schemas.py                # Output contract (DiagnosisResult, Candidate, ...) + SCHEMA_VERSION
    errors.py                 # Exception hierarchy
    parsers/                  # Log -> ParsedLog (Python traceback today)
    indexing/                 # Repo -> AST code units
    retrieval/                # ParsedLog + index -> candidate units
    ranking/                  # Candidates -> scored, ordered candidates
    inference/                # Analyzer backends (baseline, optional embedding)
    reporting/                # DiagnosisResult -> text / JSON
    evaluation/               # Reusable ranking metrics (MRR, recall@k, ...)
  tests/                      # Unit tests (mirror src modules)
tests/                        # Repo-level integration / e2e / regression tests
evaluation/                   # Benchmark harness + metrics docs + reports (honest numbers only)
datasets/                     # Data schemas + pipeline design (no large/fabricated data)
models/                       # Model configs/architecture (planned; no weights in git)
training/                     # Training pipeline (planned; never runs in CI)
examples/                     # Runnable demos + the sample fixture
docs/                         # Full specification set (start at docs/architecture.md)
tooling/, scripts/            # Maintainer tooling (prefer the Makefile)
```

## 4. Environment, build, test, lint (exact commands)

Target **Python 3.11+** (CI also runs 3.12 and 3.13). Core runtime deps are
intentionally tiny: **pydantic** and **typer** only.

```bash
# Install (editable) with dev tooling:
pip install -e "packages/python[dev]"      # or: make install

# The four quality gates (identical to CI). Run `make check` for all of them:
ruff check .                               # lint        (make lint)
ruff format --check .                      # formatting  (make format-check)
mypy                                       # strict types(make typecheck)
pytest                                     # tests       (make test)

# Handy:
make demo        # run the baseline on the bundled fixture
make build       # build sdist + wheel (then twine check)
python evaluation/run_baseline_eval.py     # honest baseline metrics
```

Optional extras (keep them optional!): `ml` (torch/transformers/sentence-transformers,
GPU-capable embedding backend), `train` (training stack), `dev` (ruff/mypy/pytest/build).

## 5. The working public surface (document-accurate)

### Python API
```python
from failurelens import FailureLens

lens = FailureLens.from_baseline()  # no deps, offline, CPU
result = lens.diagnose(
    repository="./project",
    error_log="./failure.log",  # exactly one of error_log / error_text
    top_k=5,  # optional; overrides config
)
for c in result.candidates:
    print(c.rank, c.file, c.function, c.start_line, c.end_line, c.score, c.evidence)

# Optional model backend (requires the `ml` extra + an external base model):
# lens = FailureLens.from_pretrained("org/model", revision="<pinned-sha>")
```
- Reuse an index across diagnoses: `idx = lens.index(repo); lens.diagnose(repo, error_log=..., index=idx)`.
- Errors: `InputError`, `RepositoryError`, `ModelError`, `MissingDependencyError`
  (all subclass `FailureLensError`).

### CLI (`failurelens`)
Commands: `version`, `index`, `diagnose`, `models list`, `models info`.
`--version`/`--help` work. `diagnose` key options: `--repo/-r`, `--log/-l`,
`--top/-k`, `--format/-f text|json`, `--model/-m`, `--revision`, `--quiet/-q`.

**Exit codes** (stable): `0` OK, `1` ERROR (unexpected), `3` INCONCLUSIVE
(no candidates / insufficient evidence), `4` INPUT_ERROR (bad/missing input),
`5` DEPENDENCY (missing optional extra). JSON goes to **stdout**; progress/errors
to **stderr** (use `--quiet` for clean JSON piping).

### Output schema
`DiagnosisResult` (versioned by `SCHEMA_VERSION`, currently `0.1.0`) is returned
by the API and, via `--format json`, by the CLI — **same meaning in both**. Top
fields: `schema_version, status, repository, error_type, error_message,
candidates[], analysis_method, model_id, model_revision, execution_time_ms,
warnings[]`. Each `candidate`: `rank, file, function, start_line, end_line,
score, score_type, evidence[]`. Location fields are `null` when unknown.
`status` ∈ `ok | no_candidates | insufficient_evidence | error`.

## 6. Coding conventions (match the existing code)

- **Type everything**; `mypy` runs in **strict** mode over `src/failurelens`.
- **Ruff** is the linter + formatter (config in root `pyproject.toml`; line length
  is not enforced, rules `E,F,W,I,UP,B,SIM,C4,RUF`). Run `ruff format`.
- Use `from __future__ import annotations` in library modules **except `cli.py`**
  (Typer needs runtime-evaluable annotations). Use `X | None`, not `Optional`.
- Enums that are also strings use **`StrEnum`** (Python 3.11+).
- Prefer **Pydantic** models for structured data; config models set
  `extra="forbid"`.
- Keep **core imports light**: never import torch/transformers at module top
  level in core paths — import lazily inside the `ml` backend only.
- Docstrings: concise, factual. Add comments only where they match existing
  style or explain non-obvious logic. ASCII only (ruff `RUF` flags fancy dashes).
- Determinism: ranking ties break deterministically; tests assert stable order.

## 7. How to extend (common tasks)

- **New log parser (e.g. JS/Java):** implement the parser interface in
  `parsers/`, register it, and return `ParsedLog`. Fall back gracefully on
  unsupported input (return `None`); don't crash. Add unit tests + a fixture.
- **New language indexer:** add a parser in `indexing/` producing code units with
  `start_line/end_line`; keep AST/`parse_error` fallback behaviour. Extend
  `DEFAULT_INCLUDE_SUFFIXES` only when a real indexer exists for that suffix.
- **New analysis backend:** implement the `Analyzer` protocol in `inference/`
  (`analysis_method`, `model_id`, `model_revision`, `analyze(...)`). Gate heavy
  deps behind the `ml` extra and raise `MissingDependencyError` with guidance.
  Do **not** present baseline output as a trained-model result.
- **New evaluation case:** add to `evaluation/benchmarks/*.json` with
  ground-truth `relevant_files`/`relevant_functions` and a `label_rationale`.
  Never reuse a file label as a function label.
- **New dataset record:** conform to `datasets/schemas/record.schema.json`
  (regression-tested). A valid generated case passes before and fails after the
  change; label the **root cause**, not just the surfacing site.

## 8. Security & supply chain

- Pin model **revisions**; never set `trust_remote_code=True` without reviewing
  the exact pinned code (see `docs/security.md`).
- ML extra minimum versions are security floors — do **not** lower them:
  `torch>=2.6.0`, `transformers>=5.10.0`, `sentence-transformers>=5.6.0`.
- Run `ruff`, `mypy`, `pytest` before proposing changes. CI mirrors these.
- Never commit secrets, credentials, tokens, or large binaries.

## 9. PR checklist (self-review before you finish)

- [ ] `make check` passes (ruff lint + format, mypy strict, pytest).
- [ ] New/changed behaviour has tests consistent with the existing suite.
- [ ] No fabricated models/metrics; `analysis_method`/`score_type` stay honest.
- [ ] Core deps unchanged (heavy deps stay in the `ml`/`train` extras).
- [ ] Docs/READMEs updated where the change affects them.
- [ ] No secrets, no large binaries, licenses respected.

When in doubt, prefer the **smallest correct change** and keep the baseline
runnable. See `CONTRIBUTING.md` for workflow details and `docs/` for the full
specification.
