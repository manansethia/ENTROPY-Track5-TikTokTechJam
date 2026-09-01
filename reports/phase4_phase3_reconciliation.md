# Phase 4 Step 0: Authoritative Phase 3 Numerical Reconciliation Report

*Audit Timestamp*: `2026-08-29T09:30:14Z`
*Audit Verdict*: **`PHASE_3_RECONCILED_AND_EXPLAINED`**

## 1. Apples-to-Apples Numerical Recomputation Matrix

| Model Configuration | Train Scale | Feature Dim | Val AUROC | Val AUPRC | FPR @ 0.80 | TPR @ 0.80 | FP Count | FN Count | Total Errors | Error Delta vs P2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 2 Frozen Baseline** | 82,509 | 2,212d | **0.9988** | **0.9990** | **0.87%** | **97.55%** | **37** | **149** | **186** | Baseline (0) |
| **Phase 3 Candidate B (CLIP+SigLIP)** | 20,000 | 2,176d | 0.9972 | 0.9980 | 1.56% | 96.99% | 66 | 183 | 249 | **+63 (+33.9%)** |
| **Phase 3 Candidate A (Tri-Stream)** | 20,000 | 2,212d | 0.9973 | 0.9981 | 1.70% | 96.86% | 72 | 191 | 263 | **+77 (+41.4%)** |
| **Phase 3 Candidate G (All-9 Experts)**| 20,000 | 5,130d | 0.9965 | 0.9976 | 1.72% | 96.89% | 73 | 189 | 262 | **+76 (+40.9%)** |
| **Phase 3 Candidate F (Vision+Wavelet)**| 20,000| 4,068d | 0.9969 | 0.9979 | 1.86% | 97.04% | 79 | 180 | 259 | **+73 (+39.2%)** |

## 2. Forensic Reconciliation of Narrative Contradictions

### CONTRADICTION A B C ERROR COUNT
In Phase 3, the script computed total_error_reduction = p2_total_errors (186) - total_errors (249) = -63. The markdown string template rendered '-63 fewer errors', creating the false impression of an error reduction. In truth, Candidate B (trained on 20K samples) had 249 errors, which is +63 MORE errors (+33.9% increase) than the fully-trained 82.5K Phase 2 baseline (186 errors).

### CONTRADICTION D ALL EXPERTS CLAIM
The statement 'all experts beat Phase 2 = YES' was erroneous. Under the 20K probe sweep, All-9 Experts (5,130-d) reached 0.9965 AUROC and 262 errors, which is strictly WORSE than the frozen Phase 2 baseline (0.9988 AUROC, 186 errors). Naive concatenation causes high-dimensional gradient dilution.

### CONTRADICTION E F CHAMPION NAMING
The probe sweep ranking identified B_CLIP_SigLIP_mlp2 as Rank 1 within the 20K probe challenge, but the narrative discussion referred to Gated MoE / All-Stream MoE as the long-term conceptual ideal. In reality, neither Gated MoE nor All-9 concatenation surpassed the 2-branch or 3-branch MLP on the held-out validation set under equivalent training budgets.

### APPLES TO APPLES TAKEAWAY
When compared under the EXACT SAME 20,000 training sample regime: Candidate B (2176d) had 249 errors, Candidate A Baseline (2212d) had 263 errors, Candidate G All-9 (5130d) had 262 errors. More importantly, scaling training data from 20K to 82.5K reduces Tri-Stream errors from 263 down to 186 (-29.3% error drop). Data scale and sampling quality remain the primary driver of performance.

## 3. Authoritative Architectural Hypotheses for Phase 4

- 1. Semantic Core (CLIP-ViT-L/14 + SigLIP-SO400M) provides 98%+ of total discriminative power.
- 2. Forensic / Structural Specialists (SRM-DWT, Edge-Specialist, DINOv2) provide complementary error rescue (Edge rescues 103 FNs, DINO rescues 14 FPs), but should be integrated via lightweight, gated or residual connections rather than massive 5,130-d concatenation.
- 3. 2D-FFT and Patch-MIL are noisy and redundant, adding dimensionality without unique error reduction.
- 4. Phase 4 must evaluate lightweight conditional gating (Semantic Core + Auxiliary Residuals) on fresh data.
