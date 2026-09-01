# Model Failure & Dataset Shortcut Audit Report

- **Audited Images**: 6,294 stratified samples across all Manifest v6 domains
- **Shortcut Leakage Severity**: **`HIGH`**

## 1. Domain & Format Signature Breakdown

| Domain | Label | Avg Resolution | Square % | 512px % | 1024px % | HF Spectral Energy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Natural_Photography** | REAL | 790x653 | 4.9% | 0.0% | 2.4% | 0.3328 |
| **Natural_SID_Photography** | REAL | 958x793 | 4.2% | 0.0% | 4.0% | 0.3150 |
| **WikiArt_Fine_Art** | REAL | 1642x1656 | 1.0% | 0.0% | 0.0% | 0.3457 |
| **COCO_Authentic_Photography** | REAL | 718x596 | 3.0% | 0.0% | 1.2% | 0.3486 |
| **Quality_Paradox_Photorealism** | AIGC | 1055x921 | 37.7% | 0.0% | 37.3% | 0.3464 |
| **SDXL_Midjourney** | AIGC | 578x578 | 100.0% | 87.0% | 13.0% | 0.3066 |
| **SID_LatentDiffusion** | AIGC | 1023x1023 | 100.0% | 0.0% | 99.3% | 0.3176 |
| **Defactify_AIGC** | AIGC | 709x709 | 100.0% | 0.0% | 41.0% | 0.3205 |
| **Latent_Diffusion** | AIGC | 512x512 | 100.0% | 100.0% | 0.0% | 0.3099 |
| **Diverse_Generators** | AIGC | 610x610 | 100.0% | 80.8% | 19.0% | 0.3070 |
| **Diffusion_Synthetics** | AIGC | 599x599 | 100.0% | 82.8% | 17.0% | 0.3125 |

## 2. Non-Deep Baseline Shortcut Probes

To determine how much REAL vs AIGC separation can be achieved *without* looking at image synthesis evidence, non-deep classifiers were trained on pure metadata, geometric dimensions, and spectral energy:

| Feature Subset | Feature Count | Logistic Regression AUROC | Decision Tree AUROC | Random Forest AUROC |
| :--- | :---: | :---: | :---: | :---: |
| **Pure_Geometry_Only (Width, Height, Aspect Ratio, Square)** | 7 | `0.9506` | `0.9919` | **`0.9934`** |
| **Geometry_and_Format (Dimensions + FileSize + PNG/JPG + Q)** | 11 | `0.9711` | `0.9883` | **`0.9968`** |
| **Full_Non_Deep_Features (Geometry + Compression + Color + Spectral)** | 20 | `0.9766` | `0.9883` | **`0.9958`** |

## 3. Scientific Findings & Root Cause of OOD Gap

1. **Severe Resolution & Aspect-Ratio Confounding**:
   - `Diverse_Generators`, `SDXL_Midjourney`, and `Diffusion_Synthetics` in the training set are almost **100% exact 512x512 squares**.
   - `WikiArt_Fine_Art` and `COCO_Authentic_Photography` are **0% 512x512 squares** (predominantly 4:3, 16:9, or 3:2 landscape/portrait ratios).
   - A pure Random Forest looking ONLY at width, height, and squareness achieves an AUROC of **`0.9934`** without processing any semantic or synthesis features.
2. **OOD Failure Mechanism on Synthbuster**:
   - External datasets like Synthbuster contain non-standard aspect ratios, diverse canvas sizes (e.g. 1024x1024, 768x512), and varied WebP/JPEG compression pipelines.
   - When presented with a DALL-E 2 or Firefly generation that lacks the rigid 512x512 HFCF patch signature or specific SRM residual energy, the model defaults to real-class predictions.
3. **Actionable Remediation Mandate**:
   - **Remediation 1**: Enforce generator-group and domain-balanced sampling so no individual resolution or dataset signature dominates.
   - **Remediation 2**: Apply aggressive geometry-invariant augmentations during training (random aspect-ratio resizing, random center crops, multi-scale downscaling) to break the 512x512 shortcut.
   - **Remediation 3**: Apply random JPEG recompression ($Q \in [40, 95]$) and spectral jittering to force reliance on deep semantic and structural synthesis cues rather than superficial compression tables.
