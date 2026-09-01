# Pseudo-OOD Generator & Real Domain Holdout Benchmark Report

- **Benchmark Model**: `PRODUCTION_CHAMPION_BASELINE` (Config A Frozen Checkpoint)
- **Macro-Average Pseudo-OOD Generator AUROC**: **`0.998865`**
- **Macro-Average $\text{TPR} @ 0.10\% \text{ FPR}$**: **`96.59%`**
- **Worst-Case Generator Family**: **`SID_LatentDiffusion`** ($\text{TPR} @ 0.10\% = 90.05\%$)

## 1. Epistemic Status & Scientific Distinctions

- **OBSERVED**: Zero sample or hash overlap exists between training and validation partitions ($0$ path overlap, $0$ hash overlap).
- **OBSERVED**: Performance varies significantly across generator families when tested under low-FPR operational constraints.
- **INFERRED**: Detector over-relies on resolution/aspect-ratio and high-frequency residual signatures characteristic of the dominant in-distribution training generators.
- **UNPROVEN HYPOTHESIS**: Augmentation-driven invariant fine-tuning will remediate external OOD generalization without degrading in-distribution DEV.

## 2. Generator-Family Pseudo-OOD Validation Folds

| Fold Identifier | Held-Out Generator Architecture | AIGC N | Real N | AUROC | AUPRC | Brier Score | ECE | TPR @ 0.10% FPR | TPR @ 0.01% FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold_Gen_1_SDXL_Midjourney** | SDXL_Midjourney | 681 | 5,000 | `0.999566` | `0.993814` | `0.004104` | `0.0039` | **`99.41%`** | `97.21%` |
| **Fold_Gen_2_SID_LatentDiffusion** | SID_LatentDiffusion | 613 | 5,000 | `0.997138` | `0.981427` | `0.009024` | `0.0079` | **`90.05%`** | `86.95%` |
| **Fold_Gen_3_Quality_Paradox** | Quality_Paradox_Photorealism | 1,024 | 5,000 | `0.999467` | `0.995517` | `0.004168` | `0.0037` | **`98.63%`** | `96.97%` |
| **Fold_Gen_4_Diverse_Synthetics** | Diverse_Generators & Diffusion_Synthetics | 2,682 | 5,000 | `0.999291` | `0.997974` | `0.005298` | `0.0046` | **`98.28%`** | `96.79%` |

## 3. Real-Domain Holdout Validation Folds

| Fold Identifier | Held-Out Real Domain | AIGC N | Real N | AUROC | AUPRC | Brier Score | ECE | TPR @ 0.10% FPR | TPR @ 0.01% FPR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold_Real_1_WikiArt_Fine_Art** | WikiArt_Fine_Art | 5,000 | 2,942 | `0.999565` | `0.999535` | `0.006934` | `0.0067` | **`98.30%`** | `95.68%` |
| **Fold_Real_2_COCO_Photography** | COCO_Authentic_Photography | 5,000 | 1,305 | `0.999103` | `0.999775` | `0.009286` | `0.0091` | **`97.86%`** | `97.50%` |
| **Fold_Real_3_Natural_SID_Photography** | Natural_SID_Photography | 5,000 | 503 | `0.997654` | `0.999660` | `0.010387` | `0.0104` | **`95.68%`** | `95.68%` |

## 4. Remediation Gate Standard

A candidate model (REM-A, REM-B, or REM-C) must demonstrate:
1. Measurable improvement in **Worst-Case Generator Fold TPR @ 0.10% FPR**.
2. Measurable improvement in **Macro-Average Generator Pseudo-OOD AUROC**.
3. **Zero degradation** on in-distribution DEV baseline metrics.
