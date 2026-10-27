# FailureLens

**AI-powered fault localization & developer toolkit.** Given a failure log (stack
trace) and a source repository, FailureLens ranks the code locations most likely
responsible for the failure — and shows the **evidence** for each — so developers,
CI systems, and AI coding agents can go straight to the root cause.

[![CI](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml/badge.svg)](https://github.com/LevelCapTech/FailureLens/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](docs/installation.md)

> **Honest status.** FailureLens ships a **working, dependency-light baseline**
> today (stack-trace + lexical, CPU-only, no network). It has **not** published a
> trained model yet, and makes **no** benchmark/accuracy claims that aren't
> produced by the evaluation harness. Planned work is clearly labelled as such.

## Why FailureLens?

A stack trace tells you *where an error surfaced* — not always *what caused it*.
FailureLens separates the two: it ranks candidate fault locations and attaches
only **observed** evidence (which frame, whether it's the exception site, which
error tokens matched), so you can judge each suggestion. It is **not** an LLM API
wrapper, generic code search, or an error-message summariser.

## Features

- 🔦 **Fault localization baseline** — ranks files/functions/line-spans with
  transparent, deterministic heuristics. No GPU, no network, no model download.
- 🧩 **Same result everywhere** — a single versioned schema shared by the Python
  API and the CLI's JSON output.
- 🧪 **Honest by design** — `score_type` says scores are heuristic (not
  probabilities); unknown locations are `null`, never fabricated; status
  distinguishes *inconclusive* from *failed*.
- 🪶 **Lightweight core** — only `pydantic` + `typer`. ML is an optional extra.
- 🤖 **Agent-friendly** — stable exit codes and clean JSON for automation.
- 📏 **Measurable** — reusable ranking metrics + a runnable evaluation harness.

## Install

Python **3.11+**. Not yet on PyPI — install from source:

```bash
git clone https://github.com/LevelCapTech/FailureLens.git
cd FailureLens
pip install -e "packages/python"          # baseline only (lightweight)
# pip install -e "packages/python[ml]"    # + optional embedding backend
```

See [docs/installation.md](docs/installation.md) for extras and troubleshooting.

## Quickstart

```bash
failurelens diagnose \
  --repo examples/fixtures/sample_project \
  --log  examples/fixtures/sample_project/failure.log \
  --top 5
```

```python
from failurelens import FailureLens

lens = FailureLens.from_baseline()
result = lens.diagnose(
    repository="examples/fixtures/sample_project",
    error_log="examples/fixtures/sample_project/failure.log",
    top_k=5,
)
for c in result.candidates:
    print(
        f"#{c.rank} {c.file}:{c.start_line}-{c.end_line} {c.function} "
        f"score={c.score:.3f} ({c.score_type})"
    )
    for ev in c.evidence:
        print("   -", ev.description)
```

The bundled example is instructive: the `ZeroDivisionError` **surfaces** in
`divide()`, but the **root cause** is `average()` (no empty-input guard). The
evidence shows why each candidate was selected.

## Repository layout

```
packages/python/   The installable library + `failurelens` CLI (the product)
examples/          Runnable demos + the sample fixture
evaluation/        Honest benchmark harness, metrics docs, reports
datasets/          Data record schema + pipeline design (no large/fabricated data)
models/            Model configs/architecture (planned; no weights in git)
training/          Training pipeline (planned; never runs in CI)
docs/              Full specification set (start at docs/architecture.md)
AGENTS.md          Single source of truth for contributors & AI agents
```

## Documentation

| Topic | Doc |
|-------|-----|
| How it works | [docs/architecture.md](docs/architecture.md) |
| Install | [docs/installation.md](docs/installation.md) |
| First run | [docs/getting-started.md](docs/getting-started.md) |
| Python API + schema | [docs/api.md](docs/api.md) |
| CLI + exit codes | [docs/cli.md](docs/cli.md) |
| Datasets | [docs/datasets.md](docs/datasets.md) |
| Training | [docs/training.md](docs/training.md) |
| Evaluation | [docs/evaluation.md](docs/evaluation.md) |
| Security | [docs/security.md](docs/security.md) · [SECURITY.md](SECURITY.md) |
| Model card | [docs/model-card.md](docs/model-card.md) |
| Roadmap | [docs/roadmap.md](docs/roadmap.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) · [AGENTS.md](AGENTS.md) |

## Development

```bash
make install     # editable install with dev tools
make check       # ruff lint + format check + mypy (strict) + pytest  (mirrors CI)
make demo        # run the baseline on the bundled fixture
make build       # build sdist + wheel
python evaluation/run_baseline_eval.py   # honest, measured baseline metrics
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
[AGENTS.md](AGENTS.md) (the operational contract — especially the honesty rules),
and follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Note that any *future* trained model's
license depends on its base model and training data and is **not** assumed to be
Apache-2.0 (see [docs/model-card.md](docs/model-card.md)).