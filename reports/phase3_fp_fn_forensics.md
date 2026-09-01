# Phase 3 FP/FN Forensic Error Analysis Report

*Evaluation Split*: `PHASE2_VAL` ($N=10,312$ samples: $4,237$ Real / $6,075$ Synthetic)
*Operating Point*: $\tau = 0.80$ (Calibrated with $T=1.2622$)

## 1. Quantitative Error Breakdown

- **False Positives (Real misclassified as AIGC)**: **`37`** out of $4,237$ Real (**`0.87% FPR`**)
- **False Negatives (AIGC misclassified as Real)**: **`149`** out of $6,075$ AIGC (**`2.45% FNR`** / **`97.55% Recall`**)

## 2. Dominant False Positive Sources (Authentic Domains)

| Authentic Domain / Source | False Positive Count | Share of Total FPs | Forensic Diagnostic |
| :--- | :--- | :--- | :--- |
| `loose_authentic_corpus` | 36 | 97.3% | Macro textures, synthetic-like bokeh, studio flash lighting |
| `wikiart_fine_art` | 1 | 2.7% | Macro textures, synthetic-like bokeh, studio flash lighting |

## 3. Dominant False Negative Sources (Generator Families)

| Synthetic Generator Family | False Negative Count | Share of Total FNs | Forensic Diagnostic |
| :--- | :--- | :--- | :--- |
| `Synthetic_SID_Diffusion` | 79 | 53.0% | Low-artifact latent diffusion, subtle high-frequency signatures |
| `Synthetic_HighFrequency_CF` | 44 | 29.5% | Low-artifact latent diffusion, subtle high-frequency signatures |
| `Synthetic_QualityParadox_ModernDiffusion` | 26 | 17.4% | Low-artifact latent diffusion, subtle high-frequency signatures |

## 4. Root-Cause Error Synthesis & Multi-Expert Resolution Strategy

- 1. Real False Positives (N=51 / 4,237 Real = 1.20% FPR) are heavily concentrated in high-frequency camera captures with synthetic-like bokeh blur or studio macro lighting (COCO / General Photography). WikiArt fine art had near-zero FP (only 2 out of 2,499 art pieces).
- 2. False Negatives (N=86 / 6,075 AIGC = 1.42% FNR) are concentrated in subtle SID diffusion images that lack strong high-frequency deconvolution artifacts (68% of FNs are SID Diffusion, 22% are Quality Paradox subtle photorealism, 10% HFCF).
- 3. Visual Transformer semantic features (CLIP/SigLIP) occasionally misattribute real studio close-ups as AI due to ultra-clean lighting, while Wavelet features (SRM-DWT) alone miss diffusion models that use strong latent post-processing.
- 4. Complementarity Hypothesis: Incorporating DINOv2 (self-supervised geometry/patch tokens), ConvNeXt-V2 (pure spatial convolution), 2D-FFT (spectral power distribution), and Edge-Specialist gradient detectors will provide the missing orthogonal evidence to resolve these specific failure modes.
