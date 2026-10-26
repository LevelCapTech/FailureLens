# Getting started

This walkthrough runs the FailureLens **baseline** end to end on the bundled
example fixture. It needs only the core install (no ML dependencies, no network).

## 1. Install

```bash
pip install -e "packages/python"      # or: make install  (adds dev tools)
failurelens version
```

## 2. Look at the example failure

The repository ships a tiny project whose failure is already captured:

- `examples/fixtures/sample_project/` — a 2-file project with a latent bug.
- `examples/fixtures/sample_project/failure.log` — a real `ZeroDivisionError`
  traceback.

The bug is instructive: the exception **surfaces** in `divide()`, but the **root
cause** is `average()`, which does not guard against an empty input sequence.

## 3. Diagnose from the CLI

```bash
failurelens diagnose \
  --repo examples/fixtures/sample_project \
  --log  examples/fixtures/sample_project/failure.log \
  --top 5
```

You will see ranked candidates with evidence. The baseline ranks the exception
site (`divide`) first and the root cause (`average`) second — the evidence lines
show *why* each was selected, so you can judge for yourself.

### JSON output (for tools/agents)

```bash
failurelens diagnose \
  --repo examples/fixtures/sample_project \
  --log  examples/fixtures/sample_project/failure.log \
  --format json --quiet > result.json
```

`--quiet` keeps progress messages off stdout so the JSON is clean to pipe. The
JSON conforms to the versioned schema in `docs/api.md`.

## 4. Diagnose from Python

```python
from failurelens import FailureLens

lens = FailureLens.from_baseline()
result = lens.diagnose(
    repository="examples/fixtures/sample_project",
    error_log="examples/fixtures/sample_project/failure.log",
    top_k=5,
)

print(result.status, result.error_type)
for c in result.candidates:
    print(
        f"#{c.rank} {c.file}:{c.start_line}-{c.end_line} {c.function} "
        f"score={c.score:.3f} ({c.score_type})"
    )
    for ev in c.evidence:
        print("   -", ev.description)
```

## 5. Reuse an index across runs

Indexing is the main cost on large repos. Build it once and reuse it:

```python
lens = FailureLens.from_baseline()
index = lens.index("examples/fixtures/sample_project")
r1 = lens.diagnose("examples/fixtures/sample_project", error_log="a.log", index=index)
r2 = lens.diagnose("examples/fixtures/sample_project", error_text=open("b.log").read(), index=index)
```

Pass exactly one of `error_log` (a path) or `error_text` (raw contents).

## 6. Inspect available backends

```bash
failurelens models list          # what can run here
failurelens models info          # details for the baseline
```

## Next steps

- `docs/cli.md` — full CLI reference and exit codes.
- `docs/api.md` — Python API and the output schema.
- `docs/architecture.md` — how the baseline works.
- `docs/evaluation.md` — measure accuracy honestly with the harness.
- `examples/` — more runnable examples, including agent integration.
