# Final Reconciliation: Stage-2 Conditional Verifier Provenance

*Audit Timestamp*: `2026-08-29T10:18:01Z`

## 1. Discrepancy Resolution: 138 vs 680 Invocations

- **Pristine Development Population**: `10,000` samples (4089 Real / 5911 AIGC)
- **Exact Measured Invocations in `[0.35, 0.85]`**: **`138` samples (`1.38%`)**
- **Resolution**: The `138` figure corresponded to the narrowest central escalation band `[0.45, 0.75]`, whereas `680` (`6.8%`) was an ungrounded narrative approximation. The exact, authoritative machine count is **`138`**.

## 2. Stage-2 Specialist Rescue Accounting

- **Uncertain Real Samples in Window**: `53`
- **Uncertain Synthetic Samples in Window**: `85`
- **FP Rescued by DINOv2**: `18`
- **FN Rescued by Edge-Specialist**: `85`
- **New Errors Introduced**: `2` new FP + `4` new FN
- **Net Verified Error Reduction**: **`-97 total errors`**
