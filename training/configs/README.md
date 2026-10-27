# Training configs (planned)

Experiment configuration files. A config should fully determine a run so it can
be reproduced from the file + a pinned git commit.

Expected keys (see `docs/training.md`): `base_model`, `revision`, `dataset`,
`dataset_version`, `batch_size`, `learning_rate`, `epochs`, `max_length`, `seed`,
`optimizer`, `objective`, `mixed_precision`, `output_dir`, `val_interval`,
`early_stopping`.

> Status: **planned.** No configs exist yet.
