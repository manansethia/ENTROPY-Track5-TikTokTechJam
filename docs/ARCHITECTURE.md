# Architecture

## High-level flow

```text
                         ┌───────────────────────────┐
                         │       Input RGB image     │
                         └─────────────┬─────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  │                    │                    │
                  ▼                    ▼                    ▼
        ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
        │ CLIP ViT-L/14   │  │ SigLIP          │  │ SRM high-pass       │
        │ frozen semantic │  │ frozen semantic │  │ residual filtering  │
        └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘
                 │                    │                      │
                 ▼                    ▼                      ▼
             256-d                  256-d             Haar detail bands
                 │                    │                      │
                 │                    │                      ▼
                 │                    │               ConvNeXt-Tiny
                 │                    │                      │
                 │                    │                      ▼
                 │                    │                   256-d
                 └────────────────────┼──────────────────────┘
                                      ▼
                             ┌─────────────────┐
                             │  Softmax gate   │
                             │  3 stream wts   │
                             └────────┬────────┘
                                      ▼
                             Weighted feature sum
                                      │
                                      ▼
                               2-layer MLP head
                                      │
                                      ▼
                         P(AIGC | image) ∈ [0, 1]
```

## Why this design

The supplied technical document separates synthetic-image evidence into two broad regimes: semantic/structural inconsistencies and low-level spatial-frequency fingerprints. It notes that high-frequency evidence can be strong on clean images but degrades under compression, blur and rescaling, while semantic evidence tends to survive longer.

The implementation extends that idea into three streams by using both CLIP and SigLIP semantic encoders plus a forensic residual stream.

## Parameter budget

The code does not hard-code a parameter estimate. `MasterEnsembleDetector.parameter_report()` calculates the actual instantiated total and trainable counts. This avoids accidentally violating the hackathon ceiling if a model revision has a different parameterization.

The final submission should report the exact number printed by the environment used for the final checkpoint.

## Input conventions

- Source images are read as RGB.
- The forensic stream receives an unnormalized RGB tensor in `[0, 1]`.
- CLIP input uses ImageNet-style normalization.
- SigLIP input uses mean/std `(0.5, 0.5, 0.5)`.
- All model inputs are resized to 224 × 224.

This distinction is deliberate: feeding already normalized CLIP/SigLIP values into a residual filter would make the forensic representation depend on arbitrary semantic-encoder preprocessing.

## Gating interpretation

The gate is:

```text
concat(z_clip, z_siglip, z_freq)
        ↓
Linear(768 → 128)
        ↓
GELU
        ↓
Linear(128 → 3)
        ↓
Softmax
```

The three weights sum to one. A high frequency weight is not guaranteed to mean "more trustworthy"; it simply means the learned classifier is placing more weight on that representation for that input.

## Practical caveat

The supplied design document describes a two-stream semantic/frequency gate in some sections and a tri-stream design appears in the later repository specification. This repository follows the later, concrete tri-stream implementation because it explicitly includes CLIP, SigLIP and ConvNeXt and remains under the required 2B ceiling.
