# Exact Empirical Operating Thresholds Table

*Governed Calibration Split (N=4,000: 2,000 Real, 2,000 AIGC)*

| Target Constraint | Threshold ($\tau$) | Max Allowed FP | Actual Empirical FP | Actual FPR | True Positives (TP) | Empirical TPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FPR $\le$ 1.00%** | `0.080240` | 20 | 21 | 1.050% | 1989 / 2,000 | **99.45%** |
| **FPR $\le$ 0.50%** | `0.531177` | 10 | 11 | 0.550% | 1977 / 2,000 | **98.85%** |
| **FPR $\le$ 0.10%** | `0.948647` | 2 | 3 | 0.150% | 1946 / 2,000 | **97.30%** |
| **FPR $\le$ 0.05%** | `0.953595` | 1 | 2 | 0.100% | 1945 / 2,000 | **97.25%** |
| **FPR $\le$ 0.01%** | `0.986036` | 0 | 0 | 0.000% | 1887 / 2,000 | **94.35%** |
