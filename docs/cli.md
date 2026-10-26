# CLI reference

The `failurelens` command exposes the same analysis as the Python API, with
identical JSON output. Global flags:

```bash
failurelens --help          # show commands
failurelens --version       # print version and exit
```

## Commands

### `version`
```bash
failurelens version
```
Prints the installed FailureLens version.

### `index`
```bash
failurelens index --repo ./project [--format text|json]
```
Builds a repository index and prints a summary (file/unit counts, parse errors).
Useful to check what the indexer sees before diagnosing.

### `diagnose`
```bash
failurelens diagnose --repo ./project --log ./failure.log [options]
```
Diagnoses a failure and prints ranked candidate fault locations.

| Option | Alias | Default | Meaning |
|--------|-------|---------|---------|
| `--repo` | `-r` | — | Path to the repository (**required**). |
| `--log` | `-l` | — | Path to the failure log / stack trace (**required**). |
| `--top` | `-k` | `5` | Number of candidates to return (`>= 1`). |
| `--format` | `-f` | `text` | `text` (human) or `json` (machine). |
| `--model` | `-m` | `baseline` | Analysis backend id. |
| `--revision` | | — | Pinned model revision (model backends only). |
| `--quiet` | `-q` | off | Suppress progress messages (keeps stdout clean for JSON). |

### `models`
```bash
failurelens models list [--format text|json]   # backends usable here
failurelens models info [--id <backend>] [--format text|json]   # details
```
`models list` honestly distinguishes what is **available** now (the baseline)
from external/optional entries that require the `ml` extra. No FailureLens-trained
model is listed as released, because none exists.

## Output streams

- **stdout** carries results (text or JSON).
- **stderr** carries progress and error messages.
- Use `--quiet` (or `-q`) so stdout is valid JSON for piping:

```bash
failurelens diagnose -r ./project -l ./failure.log -f json -q | jq '.candidates[0]'
```

## Exit codes

Stable and scriptable:

| Code | Name | When |
|------|------|------|
| `0` | OK | Diagnosis produced at least one candidate (`status = ok`). |
| `1` | ERROR | Unexpected error during analysis. |
| `3` | INCONCLUSIVE | Ran, but `no_candidates` or `insufficient_evidence`. |
| `4` | INPUT_ERROR | Bad or missing input (e.g. log file not found). |
| `5` | DEPENDENCY | A requested optional backend needs an extra that is not installed. |

This separates *inconclusive analysis* (`3`) from *failed execution* (`1`/`4`/`5`),
so CI and agents can react appropriately.

## JSON format

`--format json` emits the versioned `DiagnosisResult` (see `docs/api.md`). It
begins with `schema_version` and always includes `analysis_method`, so consumers
know exactly which backend produced the result and can manage compatibility.

```jsonc
{
  "schema_version": "0.1.0",
  "status": "ok",
  "repository": "examples/fixtures/sample_project",
  "error_type": "ZeroDivisionError",
  "error_message": "division by zero",
  "candidates": [
    {
      "rank": 1,
      "file": "calculator/core.py",
      "function": "divide",
      "start_line": 6,
      "end_line": 12,
      "score": 0.8,
      "score_type": "heuristic",
      "evidence": [
        {"kind": "stack_frame", "description": "Appears in stack frame 4 ...", "frame_order": 4},
        {"kind": "exception_site", "description": "Deepest in-repository frame ...", "frame_order": 4}
      ]
    }
  ],
  "analysis_method": "baseline-stacktrace-lexical",
  "model_id": null,
  "model_revision": null,
  "execution_time_ms": 1.4,
  "warnings": []
}
```

## Behavioural guarantees

- The CLI never modifies your source code and (for the baseline) never accesses
  the network.
- Missing input files produce `INPUT_ERROR` (4), not a stack trace.
- `score` is a relative heuristic, not a probability (`score_type`).
- Unknown file/line values are `null`, never fabricated.
