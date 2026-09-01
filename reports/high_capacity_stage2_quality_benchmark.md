# High-Capacity Architecture Stage 2: Fair Quality & DEV Benchmark

**Audit Date**: 2026-08-29T23:43:11Z
**Evaluation Split**: `10,000-Sample 50/50 DEV Split` (Strict Isolation: 5,000 Real / 5,000 AIGC)
**Equal Optimization Budget**: `500 Real Optimizer Steps on Identical Training Batches`

---

## 1. Stage 2 Quality Comparison Across Exact Empirical Operating Points

```
====================================================================================================
METRIC / OPERATING POINT            CONFIG A (31.9M)     CONFIG B (60.9M)     CONFIG C (90.97M)
====================================================================================================
DEV AUROC                           0.999441             0.998494             N/A (OOM)
DEV AUPRC                           0.999479             0.998634             N/A (OOM)
DEV Brier Score                     0.006931             0.024624             N/A (OOM)
DEV ECE                             0.0066               0.0271               N/A (OOM)
----------------------------------------------------------------------------------------------------
TPR @ FPR <= 1.00%                  99.28% (FP=50)       97.94% (FP=50)       N/A (OOM)
TPR @ FPR <= 0.50%                  98.92% (FP=25)       96.72% (FP=24)       N/A (OOM)
TPR @ FPR <= 0.10%                  96.02% (FP=5)       91.52% (FP=5)       N/A (OOM)
TPR @ FPR <= 0.05%                  84.20% (FP=2)       85.92% (FP=2)       N/A (OOM)
TPR @ FPR <= 0.01%                  78.46% (FP=0)       78.36% (FP=0)       N/A (OOM)
====================================================================================================
```
