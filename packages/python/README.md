# failurelens

AI-powered fault localization for Python: given a failure log (stack trace) and a
source repository, FailureLens ranks the code locations most likely responsible
for the failure, with structured, auditable evidence.

The **baseline** backend runs on CPU, needs **no network access**, and has **no
heavy dependencies** (only `pydantic` and `typer`). Model-based backends are
optional (`pip install 'failurelens[ml]'`).

> Status: early alpha. FailureLens has **not** published a trained model yet; the
> baseline is fully functional today. See the repository `docs/` for the roadmap.

## Install

```bash
pip install failurelens            # baseline only (lightweight)
pip install 'failurelens[ml]'      # + optional embedding backend
```

## Python API

```python
from failurelens import FailureLens

lens = FailureLens.from_baseline()
result = lens.diagnose(repository="./project", error_log="./failure.log", top_k=5)

print(result.status, result.analysis_method)
for candidate in result.candidates:
    print(candidate.rank, candidate.file, candidate.function, candidate.score)
    for evidence in candidate.evidence:
        print("   ", evidence.description)
```

## CLI

```bash
failurelens --help
failurelens version
failurelens index --repo ./project
failurelens diagnose --repo ./project --log ./failure.log --top 5
failurelens diagnose --repo ./project --log ./failure.log --format json
failurelens models list
```

## License

Apache-2.0. See `LICENSE`.
