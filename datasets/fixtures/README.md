# Fixtures

Tiny, checked-in example records that conform to `datasets/schemas/`. They exist
to document the format and to back the schema regression tests - they are **not**
a training corpus.

- `sample_record.json` - a synthetic record built from the in-repository
  `examples/fixtures/sample_project` (our own Apache-2.0 code, real traceback).
  It contrasts the root-cause function (`average`, positive) with the surfacing
  site (`divide`, negative).
