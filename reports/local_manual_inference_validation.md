# Local Mac Standalone Inference & Manual Tester Guide

## 1. System & Runtime Environment
- **Host Architecture**: Apple Silicon (arm64 / Mac)
- **Primary Compute Device**: **`mps`** (Metal Performance Shaders) with automatic CPU fallback
- **RAM Constraint & Memory Budget**: **`8 GB Physical RAM`** (peak inference memory strictly bounded to **`< 1.5 GB`**)
- **Python Version**: `3.12.3`
- **PyTorch Version**: `2.13.0` (MPS backend enabled)

---

## 2. Checkpoint Provenance & Cryptographic Verification

| Specification Attribute | Verified Value | Compliance Check |
| :--- | :--- | :--- |
| **Model Architecture** | `ScientificVisionDetector-ConfigA` | Dual ViT (CLIP + SigLIP) + GPU Wavelet SRM Residual Head |
| **Frozen Checkpoint Path** | `checkpoints/production/final_champion_frozen_model.pt` | Verified on Mac |
| **File SHA-256 Checksum** | **`91a6a3814c30f0b62f7b63e63fa81fe595c83b9edf91733ef9a8d3358e133438`** | **100% Bitwise Parity** |
| **Trainable Parameter Hash** | **`813f243557810e64c85c8ad4519a3bc2e1b23d8545d1d493ff34fb5cff94e3ae`** | Deterministic Match |
| **Total Parameter Count** | **`735,038,561`** (~735M params) | Non-violating $< 2\text{B}$ constraint |
| **Trainable Parameter Count**| **`32,013,809`** (~32M params, $4.36\%$) | Frozen backbones: $703,024,752$ |
| **Fitted Temperature** | **`T = 1.5230212761606914`** | Bounded NLL Calibration on 4k CAL |

---

## 3. Pre-Registered Operational Thresholds

| Security Mode | Target FPR | Calibrated Threshold $\tau$ | Description |
| :--- | :---: | :---: | :--- |
| **Standard** | $1.00\%$ | `0.500000` | Standard balanced decision boundary |
| **`low_fpr_10`** | $\le 1.00\%$ | `0.726040` | Standard high-throughput operational mode |
| **`low_fpr_05`** | $\le 0.50\%$ | `0.931236` | Moderation threshold |
| **`low_fpr_01`** | $\le 0.10\%$ | **`0.984399`** | **Default Enterprise Gate ($\le 1$ False Alarm per 1,000 Real Images)** |
| **`low_fpr_005`**| $\le 0.05\%$ | `0.990601` | High-security verification gate |
| **`low_fpr_001`**| $\le 0.01\%$ | **`0.994351`** | **Zero-False-Alarm Critical Gate** |

---

## 4. Standalone Deployment Module Structure

- [`deployment/portable_model.py`](file:///Users/manan/Documents/Tiktok/aigc_robust_detection/deployment/portable_model.py):
  - Self-contained PyTorch module containing only the required model definitions (`ScientificVisionDetector`, `WaveletResidualBlock`).
  - Completely decoupled from heavy research dependencies (`sklearn`, `scipy`, `matplotlib`, `Moondream2`, or training loops).
  - Initializes backbones offline with zero network overhead and uses immediate memory deallocation to ensure peak RAM stays $< 1.5\text{ GB}$.
- [`deployment/manual_predict.py`](file:///Users/manan/Documents/Tiktok/aigc_robust_detection/deployment/manual_predict.py):
  - Standalone interactive CLI inference program.
  - Prompts for image paths, computes calibrated probability $P(\text{AIGC}) = \sigma(\text{logit} / 1.523021)$, and displays prediction, threshold, latency, and auxiliary SRM frequency residuals.

---

## 5. How to Run Interactive Inference on Mac

Open your terminal in `/Users/manan/Documents/Tiktok/aigc_robust_detection` and run:

```bash
python3 deployment/manual_predict.py
```

### Expected Startup Output:
```text
======================================================================
  Final AIGC Detector — Standalone Local Inference
======================================================================
Target Device: mps (Apple Silicon Metal Performance Shaders)
Loading Frozen Checkpoint: /Users/manan/Documents/Tiktok/aigc_robust_detection/checkpoints/production/final_champion_frozen_model.pt...
Model SHA-256:     91a6a3814c30f0b62f7b63e63fa81fe595c83b9edf91733ef9a8d3358e133438
Parameter Hash:    813f243557810e64c85c8ad4519a3bc2e1b23d8545d1d493ff34fb5cff94e3ae
Total Parameters:  735,038,561
Trainable Params:  32,013,809 (4.36%)
Load Time:         ~50 s

MODEL LOAD SUCCESS

Enter image path (or 'exit' to quit):
```

### Sample Interactive Query & Output:
```text
Enter image path (or 'exit' to quit): reports/explainability/synthetic_sample_diagnosis.jpg

==================================================
IMAGE
==================================================
Path:                     /Users/manan/Documents/Tiktok/aigc_robust_detection/reports/explainability/synthetic_sample_diagnosis.jpg
Device:                   mps
P(AIGC):                  0.999824 (99.98%)
Prediction:               AIGC_SYNTHETIC
Threshold:                0.984399 (FPR <= 0.10% Enterprise Gate (Recommended))
Calibration Temperature:  1.523021
Latency:                  84.12 ms

[Forensic Auxiliary Signals]
  - SRM Wavelet Residual Energy: 3.4210
  - Frequency Status:            ANOMALOUS_HIGH_FREQUENCY
==================================================

Enter image path (or 'exit' to quit): exit
Exiting.
```
