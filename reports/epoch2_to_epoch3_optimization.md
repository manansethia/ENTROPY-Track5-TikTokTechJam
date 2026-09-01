# Epoch 2 to Epoch 3 Resource Optimization & Benchmark Report

**Audit Date**: 2026-08-29T16:01:15Z
**Governed Manifest**: `manifests/final_284500_governed_manifest_v6.jsonl` (`SHA: 8ec2b6916391a7e2...`)
**Evaluation Mode**: `REAL 100 WARMUP + 200 MEASURED TRAINING BATCHES WITH NUMERICALLY STABLE BCE & GRADSCALER`

---

## 1. Quantitative Benchmark Comparison Table

```
====================================================================================================
CONFIGURATION                       SAMPLES/SEC   SEC/BATCH   TORCH ALLOC   TORCH RSV    HEADROOM    LOSS        VERDICT
====================================================================================================
Epoch 2 Verified Baseline           22.14 img/s    2.168 s     3,091.2 MB    4,510.0 MB   1,427 MB    0.13467     BASELINE
Proposed BS=54 (Workers=6, PF=4)    22.86 img/s    2.3626 s   4506.6 MB    4592.0 MB   1552.0 MB    0.23234     REJECTED
Isolated BS=48 (Workers=6, PF=4)    22.48 img/s    2.1350 s   4353.6 MB    4552.0 MB   1592.0 MB    0.23354     BASELINE_PREFERRED
====================================================================================================
```

---

## 2. Decision & Selected Configuration for Epoch 3

- **Decision Verdict**: `Neither candidate achieved the mandatory +5.0% throughput improvement. Reverting to verified baseline (22.14 img/s) to maintain maximum stability.`
- **Selected Batch Size**: **`48`**
- **Selected DataLoader Workers**: **`4` persistent workers (2 threads each)**
- **Selected Prefetch Factor**: **`2`**
- **Projected Epoch 3 Duration**: **`3.06 hours`**
