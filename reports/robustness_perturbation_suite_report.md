# 8-Perturbation Robustness Suite Report

| Perturbation Condition | AUROC | AUPRC | Brier Score | ECE | TPR @ 0.10% FPR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean Baseline** | 0.999272 | 0.997343 | 0.005933 | 0.0068 | 98.59% |
| **JPEG Compression (Q=50)** | 0.997328 | 0.995499 | 0.018156 | 0.0157 | 83.94% |
| **Gaussian Blur ($\sigma=1.5$)** | 0.997544 | 0.995743 | 0.018654 | 0.0146 | 90.16% |
| **Bilinear Resize (0.5x)** | 0.997692 | 0.995862 | 0.019319 | 0.0143 | 92.77% |
| **Additive Noise ($\sigma=12$)** | 0.986592 | 0.986083 | 0.074419 | 0.0835 | 81.93% |
| **Center Crop (85%)** | 0.997764 | 0.995913 | 0.016265 | 0.0123 | 92.17% |
| **Color Jitter (0.7x)** | 0.997900 | 0.996161 | 0.015417 | 0.0123 | 94.38% |
| **Sharpening Filter** | 0.999520 | 0.997544 | 0.004849 | 0.0059 | 98.80% |
