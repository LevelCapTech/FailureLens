# Preprocessing (planned)

Normalisation and feature extraction that turns collected/generated failures into
training records: stack-trace parsing, candidate code-unit extraction,
tokenisation, and de-duplication.

Where possible, reuse the library's existing, tested components
(`failurelens.parsers`, `failurelens.indexing`) instead of re-implementing them.

> Status: **planned.**
