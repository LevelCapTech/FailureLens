# Reports

Committed evaluation reports produced by `evaluation/run_baseline_eval.py`.

Each report is a JSON file emitted by the harness. Reports are **generated
artifacts** - regenerate them by re-running the harness rather than editing by
hand:

```bash
python evaluation/run_baseline_eval.py \
  --output evaluation/reports/baseline.json
```

When committing a report, make sure it is reproducible: it should record the
`benchmark` and `analysis_method`, and the benchmark it references must be
committed too. Do not commit reports whose inputs are not in the repository.
