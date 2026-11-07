# Experiments (recorded)

Per-experiment metadata and outputs (metrics, history, config snapshot) - **not**
model weights, which are distributed separately once they exist.

Each experiment directory should capture enough to reproduce the run: git commit,
config, dataset version, seed, dependency versions, and GPU info.

Historical Stage 2 experiment metadata is stored outside Git; the recorded
experiments and their limitations are documented in the
[model card](../../docs/model-card.md). Independent model-quality evaluation
remains pending.

The optional [MLflow tracking workflow](../../docs/experiment-tracking.md)
records new runs and imports existing aggregate loss histories without uploading
private records or model weights. Local tracking state lives in the ignored
`.mlflow/` directory.
