# Architecture

FailureLens turns **failure evidence** into a ranked list of **candidate fault
locations** with auditable evidence. This document describes the pipeline and how
the code is organised.

## Overview

```
error log ──▶ [A] parse ──▶ ParsedLog ─┐
                                        ├─▶ [C] retrieve ─▶ [D] rank ─▶ [E] evidence ─▶ [F] DiagnosisResult
repository ─▶ [B] index ──▶ code units ─┘
```

The design goal is a **working baseline with no heavy dependencies**, behind
stable interfaces so smarter backends can be swapped in without changing the
contract.

## Pipeline phases

The phases map to the project issue (§8) and to Python modules:

| Phase | Responsibility | Module |
|-------|----------------|--------|
| A — Log analysis | Parse an error/stack trace into structured `ParsedLog`. | `failurelens.parsers` |
| B — Source analysis | Enumerate + parse the repository into code units (file/class/function with line spans). | `failurelens.indexing` |
| C — Candidate retrieval | Resolve stack frames to in-repo units; collect candidate units. | `failurelens.retrieval` |
| D — Ranking | Score candidates by how likely they are to be the fault. | `failurelens.ranking` |
| E — Evidence | Attach factual, observed evidence to each candidate. | `failurelens.ranking` / `failurelens.inference` |
| F — Structuring | Return a versioned `DiagnosisResult` (Python object / JSON). | `failurelens.schemas`, `failurelens.reporting` |

These are orchestrated by the **inference backend** (`failurelens.inference`) and
exposed through the **facade** (`failurelens.api.FailureLens`) and the **CLI**
(`failurelens.cli`).

## The baseline algorithm (what runs today)

The `baseline` backend is deterministic and dependency-free and has two
strategies: `stacktrace-callgraph` (`analysis_method =
"baseline-stacktrace-callgraph"`, the default, described after this list) and the
original `stacktrace-lexical` (`analysis_method = "baseline-stacktrace-lexical"`),
described here:

1. **Parse** the Python traceback into ordered `StackFrame`s (exception type,
   message, file, line, function). Frame `order` increases with depth; the
   deepest frame is where the error surfaced.
2. **Index** the repository with Python's `ast` into code units, each with a
   `start_line`/`end_line`. Files that fail to parse yield an empty unit list
   plus a recorded `parse_error` (no fabricated units).
3. **Retrieve** candidates by resolving each in-repo frame to its tightest
   enclosing unit (longest path-suffix match; unique-basename fallback).
4. **Rank** with a transparent heuristic:
   `score = 0.6·depth + 0.2·exception_site + 0.4·lexical`, where
   - `depth` rewards deeper (more specific) frames,
   - `exception_site` marks the deepest in-repo frame,
   - `lexical` counts error tokens (identifiers ≥3 chars minus stopwords, plus
     quoted literals) found in the candidate's source span (capped).
   Ties break deterministically by `(score, deepest-frame-order, file)`.
5. **Evidence**: each candidate carries only *observed* facts — which stack frame
   it came from, whether it is the exception site, and which error tokens
   matched. No LLM-generated narrative.
6. **Structure**: return a `DiagnosisResult`.

### Call-graph strategy (default)

`stacktrace-callgraph` ranks **every function and method** of the repository
(not only those in the trace):

1. The same log parsing, indexing and frame-to-unit resolution as above.
2. A static, **name-based** call graph links each function to the functions and
   methods named in its calls (`f()`, `obj.f()`), in both directions. Names shared
   by more than 8 units are ignored; dynamic dispatch and imports are not
   resolved, so unrelated functions with the same name may be linked.
3. `score = 0.25·stack + 0.25·call_relation`. `stack` uses the baseline's depth
   and exception-site formula; `call_relation` is 0.5 for one call-relation hop
   from a function in the stack and 0.25 for two. Ties break by candidate ID.
   Lexical overlap with the error message is shown as evidence only.
4. Evidence is observed and plain: the stack frames, or a `call_relation` entry such
   as "One call-relation hop (matched by name): is called by 'handle', which
   appears in stack frame 1."

The weights come from the v0.4 development experiment and were not re-tuned
(see `docs/releases/v0.5/REPORT.md` for the measurement and its limits). Very
large repositories (more than 50,000 units) skip the graph with a warning.

### Weighted call-graph strategy (opt-in, experimental)

`stacktrace-callgraph-weighted` (`analysis_method =
"baseline-stacktrace-callgraph-weighted"`) scores every function and method with a
logistic-regression decision value over 15 deterministic features: stack position,
call relations (direction and distance), same file as a stack function, the
function name appearing in the log or in the error message, and whether the
function lives in a **test file** (decided from the path only; such functions are
ranked lower). The 15 coefficients are frozen constants fitted offline on
development data (`inference/weighted_constants.py`); nothing is learned or
downloaded at run time and no model is used. It is **not** the default, and its
only measurement so far is on development data that also selected it. See
`docs/releases/v0.7/REPORT.md`.

> `score` is a **relative heuristic**, not a probability (`score_type =
> heuristic`). The baseline intentionally over-weights the exception site, so the
> *surfacing* function can outrank the *root cause*; the evidence makes this
> visible and the evaluation harness measures it honestly.

### Status semantics

- No parseable stack frames → `insufficient_evidence`.
- Frames parsed but none resolve to the repo → `no_candidates`.
- At least one candidate → `ok`.
- Unexpected failure → `error`.

## Optional embedding backend (opt-in)

`failurelens.inference.embedding` provides an **optional** reranker that uses a
sentence-transformer code encoder (via the `ml` extra). It reorders baseline
candidates by semantic similarity between the error text and candidate source.
It requires an **externally hosted** base model (nothing is bundled or
auto-downloaded) and raises `MissingDependencyError` when the extra is absent.
It is **not** a FailureLens-trained model and never presents itself as one.

## Interfaces (extension points)

- **Parsers** implement a small protocol (`parse(text) -> ParsedLog | None`) and
  are kept in a registry, so new log/trace formats (JS, Java, ...) can be added
  without touching the rest of the pipeline.
- **Indexers** produce code units from source; the Python AST indexer is the
  reference implementation.
- **Analyzers** implement the `Analyzer` protocol (`analysis_method`,
  `model_id`, `model_revision`, `analyze(...)`), so new backends plug into the
  same API/CLI and output schema.

See `AGENTS.md` §7 for step-by-step extension recipes.

## Data contract

`failurelens.schemas` is the **single source of truth** for output. It is
versioned by `SCHEMA_VERSION` and shared verbatim by the API and the CLI's JSON
output. See `docs/api.md` for field-by-field documentation.

## Non-goals (for the baseline)

- Not an LLM API wrapper, generic code search, or error-message summariser.
- Not a guarantee of fix correctness — it *suggests* candidates.
- Not multi-language yet — Python first, with interfaces ready for more (see
  `docs/roadmap.md`).
