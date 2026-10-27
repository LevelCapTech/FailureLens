# CLI examples

All commands use the bundled fixture and need only the core install.

```bash
# Human-readable diagnosis
failurelens diagnose \
  --repo examples/fixtures/sample_project \
  --log  examples/fixtures/sample_project/failure.log \
  --top 5

# Machine-readable JSON (use --quiet so stdout is clean to pipe)
failurelens diagnose \
  --repo examples/fixtures/sample_project \
  --log  examples/fixtures/sample_project/failure.log \
  --format json --quiet > result.json

# Inspect the index the analyzer builds
failurelens index --repo examples/fixtures/sample_project

# List available backends (honest about what can run here)
failurelens models list
```

Exit codes (see `docs/cli.md`): `0` ok, `1` error, `3` inconclusive,
`4` input error, `5` missing optional dependency.
