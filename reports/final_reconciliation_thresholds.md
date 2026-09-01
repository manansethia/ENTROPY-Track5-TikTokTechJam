# Final Reconciliation: Locked-Test Ultra-Low-FPR Threshold Curve

*Audit Timestamp*: `2026-08-29T10:18:04Z`

## 1. Strict Constraint Operating Frontier (Empirical FPR $\le$ Target)

| Target Constraint | Max FP Allowed | Empirical FP | Empirical FPR | Selected Threshold ($\tau$) | Empirical TPR | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\text{FPR} \le 1.00\%$ | $\le 42$ | `42` | **`0.991%`** | `tau = 0.766356` | **`97.71%`** | `99.30%` |
| $\text{FPR} \le 0.50\%$ | $\le 21$ | `21` | **`0.495%`** | `tau = 0.971936` | **`95.94%`** | `99.64%` |
| $\text{FPR} \le 0.10\%$ | $\le 4$  | `4` | **`0.094%`** | `tau = 0.999448` | **`89.93%`** | `99.93%` |
| $\text{FPR} \le 0.05\%$ | $\le 2$  | `2` | **`0.047%`** | `tau = 0.99995` | **`82.86%`** | `99.96%` |
| $\text{FPR} \le 0.01\%$ | $\le 0$  | `0` | **`0.000%`** | `tau >= 0.999967` | **`81.29%`** | `100.00%` |

> [!IMPORTANT]
> **Statistical Resolution Note**: With $N_{\text{real}} = 4,238$, a single False Positive represents $0.0236\%$. Therefore, while $0\text{ FP}$ achieves $0.00\%$ observed FPR, the test set sample size is mathematically insufficient to empirically resolve a non-zero $0.01\%$ FPR. All claims are reported with exact sample counts.
