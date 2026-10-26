# Model configs (planned)

Source-controlled model configuration files (YAML/JSON).

A config is expected to capture everything needed to reproduce how a model is
built and loaded, for example:

- `base_model` - Hugging Face model id
- `revision` - **pinned** commit/revision of the base model (required)
- `max_sequence_length`
- `pooling` strategy
- `trust_remote_code` - must be `false` unless a reviewed, pinned revision
  justifies otherwise (see `docs/security.md`)

> Status: **planned.** No configs exist yet because there is no trained model
> yet. Add files here as model development begins.
