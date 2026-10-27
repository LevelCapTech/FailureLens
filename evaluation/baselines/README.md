# Baselines (planned)

Comparison baselines for the evaluation harness. The goal is to show whether a
trained FailureLens model actually improves over simpler methods.

Planned comparison points (see `docs/evaluation.md` and the issue's §13.1):

1. **Stack-trace only** - rank candidates purely by stack-frame depth. This is
   already implemented inside the library baseline and can be isolated here.
2. **Lexical search (BM25)** - rank by lexical overlap between the error text and
   code units.
3. **Code embeddings** - rank with an off-the-shelf code retrieval model via the
   optional `ml` extra (e.g. a bi-encoder), without any fault-localization
   fine-tuning.
4. **FailureLens trained model** - once one exists; today there is **no** trained
   model and none is implied.

> Status: **planned.** This directory currently holds notes only. Add baseline
> configs/scripts here as they are implemented, and record their results with
> the harness in `evaluation/reports/`.

License note: some comparison models (e.g. SweRankEmbed-Small) are released
under non-commercial terms. Confirm and respect each model's license before
using it in evaluation; never redistribute such weights from this repository.
