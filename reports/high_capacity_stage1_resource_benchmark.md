# High-Capacity Architecture Stage 1: Resource & Memory Benchmark

**Audit Date**: 2026-08-29T23:43:11Z
**Hardware**: `Intel i5 12th Gen (12T) + NVIDIA RTX 3050 (6.14 GB VRAM) + 31 GB RAM`
**Governed Manifest**: `manifests/final_284500_governed_manifest_v6.jsonl` (`SHA: 8ec2b6916391a7e2...`)

---

## 1. Stage 1 Resource & Throughput Comparison

```
====================================================================================================
CANDIDATE CONFIG          TRAINABLE PARAMS   SAMPLES/SEC   SEC/BATCH   RESERVED VRAM   SAFE HEADROOM  STATUS
====================================================================================================
Config A (Baseline)       31.94M params        22.35 img/s   2.1470 s    4577.0 MB       1567.0 MB        SAFE (WINNER)
Config B (Mid-Scale)      60.9M params        21.18 img/s   2.2660 s    5662.0 MB       482.0 MB        REJECTED (Low Headroom)
Config C (High-Scale)     90.97M params        0.00 img/s   0.0000 s    6144.0 MB       0.0 MB        REJECTED (CUDA OOM)
====================================================================================================
```
