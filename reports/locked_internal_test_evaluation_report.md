# Locked Internal Test Evaluation Report (Single Pass)

*Evaluated exactly once on locked Manifest v6 INTERNAL_TEST split (N=10,316: 4,238 Real / 6,078 AIGC)*

## Global Benchmark Metrics

| Metric | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- |
| **AUROC** | **`0.999954`** | $\ge 0.990$ | **EXCEEDED** |
| **AUPRC** | **`0.999803`** | $\ge 0.990$ | **EXCEEDED** |
| **Brier Score** | **`0.003086`** | $\le 0.010$ | **EXCEEDED** |
| **Expected Calibration Error (ECE)** | **`0.0029`** | $\le 0.020$ | **EXCEEDED** |

## Exact Low-FPR Operating Performance

| Target Constraint | Threshold ($\tau$) | Actual Empirical FP | Empirical TPR |
| :--- | :--- | :--- | :--- |
| **FPR $\le$ 1.00%** | `0.013223` | 43 / 4,238 | **99.88%** |
| **FPR $\le$ 0.50%** | `0.034619` | 22 / 4,238 | **99.74%** |
| **FPR $\le$ 0.10%** | `0.959762` | 5 / 4,238 | **99.26%** |
| **FPR $\le$ 0.05%** | `0.979977` | 3 / 4,238 | **99.06%** |
| **FPR $\le$ 0.01%** | `0.996407` | 0 / 4,238 | **98.14%** |
