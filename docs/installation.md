# Installation

FailureLens targets **Python 3.11+** (CI also exercises 3.12 and 3.13). The core
package is intentionally lightweight — its only runtime dependencies are
**pydantic** and **typer** — so the baseline installs quickly and runs offline on
CPU.

> The package is not yet published to PyPI. Until then, install from source as
> shown below. The PyPI name `failurelens` is a placeholder pending availability
> confirmation.

## From source (current method)

```bash
git clone https://github.com/LevelCapTech/FailureLens.git
cd FailureLens
pip install -e "packages/python"            # core baseline only
```

Verify it works:

```bash
failurelens version
failurelens diagnose \
  --repo examples/fixtures/sample_project \
  --log examples/fixtures/sample_project/failure.log \
  --top 5
```

## Optional extras

Extras are kept optional so the baseline never pulls heavy dependencies:

| Extra | Installs | Use for |
|-------|----------|---------|
| `dev` | ruff, mypy, pytest, build | Development & CI quality gates |
| `ml` | torch, transformers, sentence-transformers, huggingface-hub, numpy | Optional embedding reranker backend |
| `train` | the `ml` stack plus training/eval helpers | Future model training (GPU) |

```bash
pip install -e "packages/python[dev]"       # contributors
pip install -e "packages/python[ml]"        # embedding backend (downloads a base model on first use)
```

The `ml` extra pins **security floors** (`torch>=2.6.0`, `transformers>=5.10.0`,
`sentence-transformers>=5.6.0`). Do not lower these (see `docs/security.md`).

## Using the Makefile

```bash
make install      # pip install -e "packages/python[dev]"
make check        # ruff + format check + mypy + pytest (mirrors CI)
make demo         # run the baseline on the bundled fixture
```

## Offline / air-gapped use

The **baseline requires no network access** and downloads nothing. Only the
optional `ml`/`train` backends fetch an external base model, and only when you
explicitly request one. Nothing is downloaded implicitly.

## Troubleshooting

- **`failurelens: command not found`** — ensure your environment's `bin/` is on
  `PATH`, or run `python -m failurelens.cli`.
- **`MissingDependencyError` / exit code 5** — you requested a model backend
  without the `ml` extra; install it or use the baseline.
- **Python too old** — install Python 3.11 or newer.
