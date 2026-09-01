# Phase 7 Conditional Verifier Provenance Audit Report

*Audit Timestamp*: `2026-08-29T10:12:50Z`

## 1. Executive Reconciliation

- **Population**: 10,000 development images (4089 Real / 5911 AIGC)
- **Uncertainty Routing Window**: `[0.35, 0.85]`
- **Stage 2 Invocations**: 138 images (**`1.38%`** of test population)
- **False Positives Rescued**: **`18`** macro/bokeh false alarms pulled below $\tau=0.80$
- **False Negatives Rescued**: **`112`** subtle latent diffusion missed fakes pushed above $\tau=0.80$
- **New Errors Introduced**: `2` new FP + `4` new FN
- **Net Error Reduction**: **`-124 total errors`** (FP dropped from 35 to 19; FN dropped from 142 to 34)
- **AUROC Improvement**: `0.9990` $\to$ **`0.9994`** (+0.0004 gain)
