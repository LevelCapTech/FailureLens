# Agent integration

FailureLens is designed to be called by AI coding agents and CI. The pattern:

1. The agent runs `failurelens diagnose ... --format json --quiet`.
2. It parses the JSON (`schema_version`, `candidates[]` with `file`, `function`,
   `start_line`/`end_line`, `score`, `evidence[]`).
3. It opens the top candidates, reads the cited evidence, and proposes a fix.
4. It re-runs the failing test to confirm.

Key properties that make this safe to automate:

- Stable **exit codes** separate *inconclusive* (`3`) from *failed execution*
  (`1`/`4`/`5`).
- JSON goes to **stdout**, progress/errors to **stderr** (`--quiet` keeps stdout
  clean).
- Unknown locations are `null`, never fabricated; `score_type` marks scores as
  heuristic, not probabilities.
- The baseline makes **no network calls** and never modifies your code.

See `agent_diagnose.py` for a minimal, dependency-free driver that shells out to
the CLI and prints the top candidates as a compact summary.
