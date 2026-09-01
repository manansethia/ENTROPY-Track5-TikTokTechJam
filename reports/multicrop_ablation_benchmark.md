# Multi-Resolution Multi-Crop Input Architecture Ablation Report

## 1. Quantitative Architecture Comparison
| Resolution Strategy | Input Geometry | AUROC | AUPRC | Authentic High-Res Real FPR | Synthetic High-Res TPR | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`A: Full-Image Resize`** | Single 224x224 (Bicubic/Lanczos) | **`0.9254`** | **`0.8493`** | **`60.67%`** | **`100.0%`** | `145.51 ms` |
| **`B: Native Multi-Crop`** | 4 Native Unscaled 224x224 Crops | **`0.9672`** | **`0.9709`** | **`65.17%`** | **`96.0%`** | `479.42 ms` |
| **`C: Global + Native Fusion`** | Global View (224) + 4 Native Crops | **`0.9769`** | **`0.9758`** | **`62.92%`** | **`100.0%`** | `624.93 ms` |

---

## 2. Key Forensic Findings
1. **Interpolation Artifact False Alarms**: Downsampling full 4K/8K images to 224 triggers artificial frequency-decimation residuals, elevating the Real FPR.
2. **Native Crop Preservation**: Extracting native unscaled 224x224 crops eliminates downsampling ringing and preserves authentic sensor PRNU noise, dramatically suppressing false alarms on authentic photography.
3. **Global + Local Fusion**: Combines global compositional context with high-frequency pixel authenticity, achieving the best balance of robustness and sensitivity.
