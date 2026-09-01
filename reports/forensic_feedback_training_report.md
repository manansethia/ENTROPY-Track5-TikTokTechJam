# Forensic Explanation Feedback Training Report

- **Champion Architecture**: Config A (31.94M Trainable Parameters)
- **Starting Checkpoint**: `/home/manan/aigc_robust_detection/checkpoints/high_capacity/candidate_config_A.pt`
- **Selected Final Stage**: `ROUND_1`

## Comparative DEV Progression

| Stage | AUROC | AUPRC | Brier Score | ECE | TPR @ 0.10% FPR | TPR @ 0.01% FPR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Feedback Baseline** | 0.999431 | 0.999271 | 0.007073 | 0.0067 | 97.24% | 79.28% |
| **Feedback Round 1** | 0.999430 | 0.999270 | 0.007262 | 0.0068 | 97.30% | 79.42% |
| **Feedback Round 2** | 0.999430 | 0.999271 | 0.007041 | 0.0065 | 97.24% | 78.98% |

- **Feedback Delta Verification**: Confirmed parameter hash transitions across both rounds with $L_2$ gradient norm updates.
- **Rollback Rule Decision**: Model `ROUND_1` selected as champion.
