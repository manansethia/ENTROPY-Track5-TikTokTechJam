# Post-Hoc Temperature Scaling & Calibration Report

- **Calibration Dataset Split**: 4,000 Samples (2,000 Real / 2,000 AIGC, Manifest v6 CAL split)
- **Optimal Scaled Temperature ($T^*$)**: `1.5695`

## Calibration Metric Reductions

| Metric | Raw Logits ($T=1.0$) | Temperature-Scaled Logits ($T=1.5695$) | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Brier Score** | 0.006974 | 0.006460 | 7.37% reduction |
| **Expected Calibration Error (ECE)** | 0.0067 | 0.0048 | 27.84% reduction |
