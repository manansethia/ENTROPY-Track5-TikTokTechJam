# Final standalone student design

The final release candidate will be a single image model. Teacher checkpoints
are used only during training-time supervision and are not part of its runtime.

The proposed architecture targets roughly 90–140M parameters: a visual
backbone with a feature pyramid, a learned high-pass branch seeded with fixed
SRM filters, frequency-domain summaries, cross-scale global/local fusion, a
three-class head (`REAL`, `PARTIAL_AI`, `FULL_AIGC`), and a decoder producing a
localization probability map. Affected area will be derived from the decoder
output after an explicitly evaluated threshold, never from image-level score.

Ground-truth class and mask supervision are primary. Only audited, reproducible
teachers may provide auxiliary logits, features, or spatial targets. The final
training manifest must have source-lineage grouping and an independent split;
the protected organizer benchmark is excluded from every training-time stage.

Manan Sethia — project author
