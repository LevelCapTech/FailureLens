# Training configs (planned)

Experiment configuration files. A config should fully determine a run so it can
be reproduced from the file + a pinned git commit.

Expected keys (see `docs/training.md`): `base_model`, `revision`, `dataset`,
`dataset_version`, `batch_size`, `learning_rate`, `epochs`, `max_length`, `seed`,
`optimizer`, `objective`, `mixed_precision`, `output_dir`, `val_interval`,
`early_stopping`.

> Status: `stage2_config.json` and `toy_config.json` are used by the existing
> Stage 2 and toy pipelines. `stage2_v02_{a,b,c}.json` are v0.2 candidate runs
> over the private corpus described in `docs/corpus-v02.md`; they reference it
> by digest only (`PENDING_CORPUS_BUILD` until the corpus is built).
