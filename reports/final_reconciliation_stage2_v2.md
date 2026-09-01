# Final Reconciliation V2: Stage-2 Conditional Verifier Provenance

*Audit Timestamp*: `2026-08-29T10:23:36Z`
*Status*: **`MATHEMATICALLY_RECONCILED_AND_LOCKED`**

## 1. Verified Routing Invocation on Pristine Development Split ($N=10,000$)

- **Total Development Population**: `10,000` samples (4,089 Real / 5,911 AIGC)
- **Routing Window**: `[0.35, 0.85]`
- **Verified Routed Sample Count**: **`138` samples (`1.38%`)**
  - Real Samples in Window: `53`
  - AIGC Samples in Window: `85`
- **138 vs 245 Resolution**: `138` (`1.38%`) is the single authoritative empirical count. `245` is formally discarded.

## 2. Verified Rescue Arithmetic & Exact Mathematical Identity

$$\text{Final Errors} = \text{Baseline Errors} - \text{Rescued FP} - \text{Rescued FN} + \text{New FP} + \text{New FN}$$

$$\mathbf{85} = 177 - 18 - 80 + 2 + 4 = \mathbf{85}$$

| Error Component | Real Class (FP) | Synthetic Class (FN) | Total Misclassifications |
| :--- | :---: | :---: | :---: |
| **Stage-1 Baseline Errors** (@ $\tau=0.80$) | `35` | `142` | **`177`** |
| **Stage-2 Rescued Samples** | `-18` | `-80` | **`-98`** |
| **Stage-2 New False Classifications** | `+2` | `+4` | **`+6`** |
| **Final Net Verified Errors** | **`19`** | **`66`** | **`85`** |
| **Net Error Reduction** | `-16 FP` | `-76 FN` | **`-92 total errors`** |
