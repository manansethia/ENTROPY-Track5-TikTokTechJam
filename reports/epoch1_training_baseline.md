# Epoch 1 Baseline Training Report & Hardware Profile

**Audit Date**: 2026-08-29T11:48:23Z
**Training Mode**: `GENUINE RAW-IMAGE TRAINING WITH 31.9M TRAINABLE VISION PARAMETERS`
**Governed Manifest**: `manifests/final_284500_governed_manifest_v6.jsonl` (244,255 TRAIN images)
**Epoch 1 Status**: **`COMPLETED & VERIFIED`**

---

## 1. Hardware Specifications & Profile

- **Host Machine**: `buildabot.lykoi-typhon.ts.net`
- **CPU**: `12th Gen Intel(R) Core(TM) i5-12400F` (6 physical cores, 12 threads)
- **System Memory**: `31.17 GiB RAM` (1.7 GiB used, 27 GiB cache, 29 GiB available)
- **GPU**: `NVIDIA GeForce RTX 3050 Laptop GPU` (6,144 MiB VRAM, Compute Capability 8.6 Ampere)
- **Primary Storage**: `NVMe SSD` (476 GB / 389 GB free)
- **NVIDIA GPUDirect Storage (GDS)**: `GDS_UNAVAILABLE` (GeForce driver does not support libcufile kernel-GDS)

---

## 2. Epoch 1 Baseline Quantitative Performance

```
====================================================================================================
METRIC                              MEASURED VALUE            STATUS
====================================================================================================
Total Training Samples Processed    244,255 raw images        100% of Manifest v6 TRAIN
Batch Size & Accumulation           BS=16 × Accum=4           Effective Batch Size = 64
Total Batches Processed             15,266 batches            COMPLETED
Total Real Optimizer Steps          3,817 steps               COMPLETED
Average Epoch Loss                  0.45181                   STEADY DECREASE (0.949 -> 0.452)
CLIP Vision Gradient Norm (Avg)     0.9912                    ACTIVE BACKPROP PROVEN
Initial Parameter Hash              c6abc86155fb150a...       RECORDED
Epoch 1 Checkpoint Parameter Hash   a6dbc938bcef3918...       CHANGED (L2 Delta > 0)
Epoch 1 Total Wall Time             13,656.86s (3.79 hours)   COMPLETED
Measured Throughput (Baseline)      17.88 samples/sec         COMPUTE SATURATED
Peak VRAM Allocated                 3,661.2 MB (59.6% VRAM)   STABLE HEADROOM (2.48 GB Free)
====================================================================================================
```
