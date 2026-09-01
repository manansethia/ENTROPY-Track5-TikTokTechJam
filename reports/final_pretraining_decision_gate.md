# Master Protocol Final Pre-Training Decision-Gate Report

*Date & Timestamp: 2026-08-28 18:33:15Z*  
*Authoritative Status: **PRE-TRAINING DECISION GATE HALTED FOR HUMAN REVIEW***  
*Report JSON: [`reports/final_pretraining_decision_gate.json`](file:///Users/manan/Documents/Tiktok/aigc_robust_detection/reports/final_pretraining_decision_gate.json)*

---

## 1. Frozen Provenance & Integrity Verification

* **Master Dataset Manifest**: [`manifests/fresh_5k_manifest.jsonl`](file:///Users/manan/Documents/Tiktok/aigc_robust_detection/manifests/fresh_5k_manifest.jsonl)
* **Cryptographic SHA-256 Hash**: `890bd3c373673e3c0b2eb92abb7d3fdfb43984a00327a6a551cc592f7e3f3467`
* **Random Sampling Seed**: `20260828`
* **Active Split Sample Counts**:
  * **FRESH_TRAIN**: `1,000` images ($500\text{ Real} / 500\text{ Fake}$) — Used strictly for linear probe & fusion fitting.
  * **FRESH_VAL**: `300` images ($150\text{ Real} / 150\text{ Fake}$) — Used for 7-perturbation validation and complementarity audit.
  * **FRESH_INTERNAL_TEST**: `500` images ($245\text{ Real} / 255\text{ Fake}$) — Strictly untouched, held-out validation.
* **External Benchmarks Quarantined**: `Synthbuster`, `AIGIBench`, `Chameleon`, `VCT²`, `WildRF`, `SynthWildX` remain 100% locked.

---

## 2. Fresh Validation & Untouched Internal-Test Audits

```
=============================================================================================================================================================
CROSS-SPLIT PERFORMANCE AUDIT (VALIDATION VS UNTOUCHED INTERNAL TEST)
=============================================================================================================================================================
Candidate Architecture                  Params    Val AUROC  Val AUPRC   Val FPR [95% CI]        Test AUROC Test AUPRC  Test FPR [95% CI]      Test ECE
-------------------------------------------------------------------------------------------------------------------------------------------------------------
[BASELINE] CLIP-ViT-L (Single)          427.6M     0.9783     0.9814     8.0% [4.4% - 13.9%]       0.9785     0.9806     6.5% [3.9% - 10.5%]     0.4735
1. CLIP + SigLIP (Learned Logistic)    1305.0M     0.9857     0.9894     3.3% [1.3% - 7.9%]        0.9828     0.9850     4.1% [2.1% - 7.5%]      0.4705
2. CLIP + SigLIP + SRM-DWT (Wavelet)   1305.0M     0.9854     0.9891     2.7% [0.9% - 7.0%]        0.9829     0.9852     3.7% [1.8% - 7.0%]      0.4691
3. CLIP + SigLIP + DINOv2 (Tri-Vision) 1609.3M     0.9845     0.9882     4.0% [1.7% - 8.9%]        0.9826     0.9848     4.5% [2.4% - 8.0%]      0.4718
4. Quad-Expert (CLIP+SigLIP+DINO+SRM)  1609.3M     0.9843     0.9879     5.3% [2.6% - 10.5%]       0.9824     0.9846     4.9% [2.7% - 8.5%]      0.4712
5. CLIP + SigLIP (Simple Avg)          1305.0M     0.9826     0.9865     2.7% [0.9% - 7.0%]        0.9804     0.9829     4.1% [2.1% - 7.5%]      0.4578
6. CLIP + 2D-FFT + SRM-DWT (Triad)      427.6M     0.9802     0.9834     6.7% [3.5% - 12.3%]       0.9791     0.9812     5.7% [3.3% - 9.5%]      0.4741
7. CLIP + DINOv2 (Cross-Attention)      732.0M     0.9795     0.9835     5.3% [2.6% - 10.5%]       0.9790     0.9810     5.3% [3.0% - 9.0%]      0.4715
=============================================================================================================================================================
```

---

## 3. False Positive Rate (FPR) Statistical Uncertainty & Threshold Sweeps

*Evaluating candidate **`CLIP + SigLIP + SRM-DWT`** across operational thresholds on the **500-sample Untouched Internal Test** ($N_{real}=245, N_{fake}=255$):*

| Decision Threshold ($	au$) | True Negatives ($TN$) | False Positives ($FP$) | False Negatives ($FN$) | True Positives ($TP$) | FPR (%) | Wilson 95% Confidence Interval | Precision (%) | FNR (%) | Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$	au = 0.50$** | 236 | 9 | 19 | 236 | **3.67%** | **[1.80%, 7.00%]** | 96.33% | 7.45% | 94.40% |
| **$	au = 0.60$** | 239 | 6 | 23 | 232 | **2.45%** | **[1.00%, 5.40%]** | 97.48% | 9.02% | 94.20% |
| **$	au = 0.70$** | 241 | 4 | 28 | 227 | **1.63%** | **[0.50%, 4.30%]** | 98.27% | 10.98% | 93.60% |
| **$	au = 0.80$** | 243 | 2 | 34 | 221 | **0.82%** | **[0.15%, 3.10%]** | 99.10% | 13.33% | 92.80% |
| **$	au = 0.85$** | 244 | 1 | 41 | 214 | **0.41%** | **[0.05%, 2.40%]** | 99.53% | 16.08% | 91.60% |
| **$	au = 0.90$** | 245 | 0 | 50 | 205 | **0.00%** | **[0.00%, 1.60%]** | 100.00% | 19.61% | 90.00% |
| **$	au = 0.95$** | 245 | 0 | 66 | 189 | **0.00%** | **[0.00%, 1.60%]** | 100.00% | 25.88% | 86.80% |

---

## 4. Error Complementarity & Bilateral Rescues

* **`CLIP-ViT-L` vs `SigLIP-SO400M`**:
  * Pearson Correlation: `0.78` | Disagreement: `10.7%` | Oracle AUROC: **`0.9944`**
  * Rescues: `CLIP` rescues **20 errors** of `SigLIP`; `SigLIP` rescues **15 errors** of `CLIP`.
* **`CLIP-ViT-L` vs `SRM-DWT-Wavelet`**:
  * Pearson Correlation: `0.32` | Disagreement: `40.3%` | Oracle AUROC: **`0.9975`**
  * Rescues: `CLIP` rescues **106 errors** of `SRM-DWT`; `SRM-DWT` rescues **15 errors** of `CLIP`.
* **`CLIP-ViT-L` vs `DINOv2-Registers`**:
  * Pearson Correlation: `0.58` | Disagreement: `24.0%` | Oracle AUROC: **`0.9912`**
  * Rescues: `CLIP` rescues **39 errors** of `DINOv2`; `DINOv2` rescues **16 errors** of `CLIP`.

---

## 5. Marginal Decomposition of Expert Contributions

1. **`CLIP-ViT-L/14` (Core Foundation)**:
   * Provides rapid ($79.1\text{ms}$), high-level semantic discrimination ($0.9785\text{ Test AUROC}$) and excellent unperturbed baseline accuracy.
2. **`+ SigLIP-SO400M` (Dual-VLM Diversity)**:
   * Contributes independent pretraining objectives (Sigmoid BCE vs InfoNCE Softmax), reducing False Positive Rate from $6.5\%$ to $4.1\%$ and yielding $+0.0044\text{ Test AUROC}$ gain.
3. **`+ SRM-DWT Wavelet Residuals` (Forensic High-Pass Channel)**:
   * Adds zero parametric bloat ($0.01\text{M}$ parameters, $+1.0\text{ms}$ latency) while capturing high-frequency score matching and deconvolution Fourier artifacts, cutting FPR to the minimum observed ($2.7\%\text{ Val} / 3.7\%\text{ Test}$).
4. **`+ DINOv2-Registers` (Self-Supervised Structural Vision)**:
   * Enhances perturbation floor on extreme downscaling and defocus blur ($+0.0420\text{ Worst-Case AUROC}$), but increases memory footprint by $+304\text{M}$ parameters and adds $+82\text{ms}$ latency.

---

## 6. Authoritative Decision & Recommendation

### Recommended Champion Architecture:
**Candidate B: `CLIP-ViT-L/14` + `SigLIP-SO400M-224` + `SRM-DWT Wavelet Head`**
* **Total Instantiated Parameters**: **`1,304.98 Million`** ($< 2,000,000,000$ competition budget).
* **Peak GPU VRAM**: **`3.70 GB`** on NVIDIA RTX 3050 ($< 6.0\text{ GB}$ limit).
* **Inference Latency**: **`185.1 ms`** per sample.
* **Test AUROC**: **`0.9829`** | **Test AUPRC**: **`0.9852`** | **Test FPR**: **`3.67%`** (Wilson 95% CI: $[1.80\%, 7.00\%]$).
* **High-Precision Operating Point**: At $\tau = 0.80$, $\text{FPR} = 0.82\%$ ($[0.15\%, 3.10\%]$) with $99.1\%$ Precision.

---

## 7. Next Step: Formal Section 30 Approval

Large-scale multi-GB dataset training and feature caching remain strictly stopped. Upon your approval of Candidate B (or Candidate A/C), we will advance to **Section 12 (Large-Scale Multi-GB Manifest Construction from approved sources on `/mnt/ai-storage`)** and **Section 13 (Supervised Training)**.
