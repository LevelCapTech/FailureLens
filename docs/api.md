# Python API

The public API is small and stable. Import everything from the top-level package.

```python
from failurelens import (
    FailureLens,  # the facade
    DiagnosisResult,
    Candidate,
    Evidence,
    StackFrame,
    ParsedLog,  # output models
    AnalysisStatus,
    ScoreType,
    EvidenceKind,  # enums
    IndexConfig,
    DiagnoseConfig,  # configuration
    FailureLensError,
    InputError,
    RepositoryError,
    ModelError,
    MissingDependencyError,
    SCHEMA_VERSION,
    __version__,
)
```

## `FailureLens`

### Constructors

```python
FailureLens.from_baseline(*, parser_registry=None) -> FailureLens
```
Dependency-free baseline backend. Runs on CPU, offline. **This is the default.**

```python
FailureLens.from_pretrained(model_id, *, revision=None, device=None, parser_registry=None) -> FailureLens
```
Optional embedding backend. Requires the `ml` extra and an **externally hosted**
base model (nothing is bundled or auto-downloaded). Pin `revision` for
reproducibility. Raises `MissingDependencyError` if the extra is missing. This is
**not** a FailureLens-trained model.

### Methods

```python
lens.index(repository, *, config: IndexConfig | None = None) -> RepositoryIndex
```
Builds a reusable repository index. Pass the result to `diagnose(..., index=...)`
to avoid re-scanning.

```python
lens.diagnose(
    repository,
    *,
    error_log=None,        # path to a log file           (exactly one of
    error_text=None,       # raw log contents              these two)
    top_k=None,            # override config.top_k
    index=None,            # reuse a prebuilt index
    config=None,           # DiagnoseConfig
) -> DiagnosisResult
```
Diagnoses a failure and returns ranked candidates. Provide **exactly one** of
`error_log` or `error_text` (otherwise `InputError`).

### Properties

- `lens.analyzer` — the active backend (`analysis_method`, `model_id`, `model_revision`).
- `lens.parsers` — the parser registry.

## Configuration

### `IndexConfig`
| Field | Default | Meaning |
|-------|---------|---------|
| `include_suffixes` | `(".py",)` | File suffixes to index. |
| `exclude_dirs` | common vcs/venv/cache dirs | Directory names pruned during traversal. |
| `exclude_globs` | `()` | `fnmatch` patterns (repo-relative POSIX) to exclude. |
| `max_file_bytes` | `1_000_000` | Skip files larger than this (`0` disables). |
| `follow_symlinks` | `False` | Whether to follow symlinks. |

### `DiagnoseConfig`
| Field | Default | Meaning |
|-------|---------|---------|
| `top_k` | `5` | Max candidates to return. |
| `index` | `IndexConfig()` | Indexing options. |
| `read_candidate_source` | `True` | Read candidate spans to compute lexical evidence. |

Both models set `extra="forbid"`, so unknown keys raise.

## Output schema

`DiagnosisResult` is the contract shared by the API and the CLI's JSON output. It
is versioned by `SCHEMA_VERSION` (currently `0.1.0`); JSON output includes
`schema_version` so consumers can manage compatibility.

### `DiagnosisResult`
| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | `str` | Equals `SCHEMA_VERSION`. |
| `status` | `AnalysisStatus` | `ok` / `no_candidates` / `insufficient_evidence` / `error`. |
| `repository` | `str \| None` | Path analyzed. |
| `error_type` | `str \| None` | Exception/error class. |
| `error_message` | `str \| None` | Error message. |
| `candidates` | `list[Candidate]` | Ranked, best first. |
| `analysis_method` | `str` | e.g. `baseline-stacktrace-lexical`. |
| `model_id` | `str \| None` | Set only for model backends. |
| `model_revision` | `str \| None` | Pinned revision, if any. |
| `execution_time_ms` | `float` | Wall-clock analysis time. |
| `warnings` | `list[str]` | Non-fatal notes. |

Helper: `result.to_json(indent=2) -> str`.

### `Candidate`
| Field | Type | Notes |
|-------|------|-------|
| `rank` | `int` | 1-based; 1 is most likely. |
| `file` | `str \| None` | Repo-relative POSIX path, or `None`. |
| `function` | `str \| None` | Enclosing function/method, or `None`. |
| `start_line` / `end_line` | `int \| None` | 1-based span, or `None`. |
| `score` | `float` | Relative ranking score (see `score_type`). |
| `score_type` | `ScoreType` | Currently `heuristic` (**not** a probability). |
| `evidence` | `list[Evidence]` | Observed supporting facts. |

> Location fields are `None` when they cannot be determined — never a guessed
> value.

### `Evidence`
| Field | Type | Notes |
|-------|------|-------|
| `kind` | `EvidenceKind` | e.g. `stack_frame`, `exception_site`, `lexical_match`. |
| `description` | `str` | Short, factual description. |
| `frame_order` | `int \| None` | Related stack-frame index, if any. |

### `StackFrame` / `ParsedLog`
`ParsedLog` holds `parser`, `language`, `error_type`, `error_message`, and
`frames: list[StackFrame]`. Each `StackFrame` has `order`, `file`, `line`,
`function`, `in_repository`, `repository_path`.

## Errors

All inherit from `FailureLensError`:

- `InputError` — invalid/missing input (e.g. not exactly one log source).
- `RepositoryError` — repository could not be read/indexed.
- `ModelError` — a model backend failed.
- `MissingDependencyError` — an optional extra (e.g. `ml`) is not installed.

## Example

```python
from failurelens import FailureLens, AnalysisStatus

lens = FailureLens.from_baseline()
result = lens.diagnose(
    repository="examples/fixtures/sample_project",
    error_log="examples/fixtures/sample_project/failure.log",
)

if result.status is AnalysisStatus.OK:
    top = result.candidates[0]
    print(top.file, top.function, top.score)
else:
    print("inconclusive:", result.status, result.warnings)
```
