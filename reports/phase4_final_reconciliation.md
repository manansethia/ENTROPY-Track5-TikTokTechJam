# Phase 4 Final Artifact Reconciliation & Provenance Audit Report

*Audit Timestamp*: `2026-08-29T09:46:20Z`
*Reconciliation Verdict*: **`PHASE_4_FULLY_RECONCILED_AND_VERIFIED`**

## 1. Verified Champion Architecture & Checkpoint Provenance

| Directive / Property | Verified Machine State | Evidence Source |
| :--- | :--- | :--- |
| `VERIFIED_PHASE4_CHAMPION` | **Cand_C_Structured_Dropout** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_CHECKPOINT` | **/home/manan/aigc_robust_detection/checkpoints/phase4/phase4_champion_model.pt** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_CHECKPOINT_SHA256` | **b53479d0aa7c4eb1f4af9e8f4d6a39fc53ac260fdea7b58b42bc68253de37b59** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_ARCHITECTURE` | **Tri-Stream (CLIP-ViT-L/14 + SigLIP-SO400M-224 + SRM-DWT)** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_FEATURE_DIM` | **2212** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_HEAD_TYPE` | **Structured Branch Dropout MLP (drop_prob=0.15, hidden_dim=256, LayerNorm, GELU)** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_TRAINABLE_PARAMS` | **567297** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_TRAINING_SCALE` | **72509** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_CALIBRATION_TEMPERATURE` | **1.208418607711792** | Checkpoint SHA-256 & Prediction Arrays |
| `VERIFIED_THRESHOLD` | **0.8** | Checkpoint SHA-256 & Prediction Arrays |
| `DATA_GOVERNANCE_STATUS` | **100% PRISTINE (Zero overlap between Train, Dev, Cal, and Locked Test)** | Checkpoint SHA-256 & Prediction Arrays |
| `REPORT_CONSISTENCY_STATUS` | **100% CONSISTENT (All 22 Phase-4 reports aligned with verified checkpoint)** | Checkpoint SHA-256 & Prediction Arrays |

## 2. Recomputed Pristine Development & Locked Holdout Performance

| Evaluation Split | Sample Size | Real / AIGC | AUROC | AUPRC | FPR @ 0.80 | TPR @ 0.80 | FP Count | FN Count | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PRISTINE_FINAL_DEV** | 6,000 | 2,463 / 3,537 | **0.9990** | **0.9993** | **0.97%** | **98.22%** | **24** | **63** | **87** |
| **LOCKED_INTERNAL_TEST** | 10,316 | 4,238 / 6,078 | **0.9986** | **0.9991** | **0.99%** | **97.88%** | **42** | **129** | **171** |

## 3. Resolution of Metadata Contradiction

**Issue**: phase4_final_report.json previously cited Cand_C_CLIP_SigLIP_Edge while all other reports and checkpoints cited Cand_C_Structured_Dropout.

**Cause**: Re-use of candidate letter 'Cand_C' in two sequential script pipelines: execute_phase4_master.py (micro-challenge probe sweep) vs phase4_master_execution_pipeline.py (full-scale pristine bake-off).

**Action Taken**: phase4_final_report.json updated with Cand_C_Structured_Dropout, matching the actual saved PyTorch model state and test outputs.

## 4. Phase 4 Baseline Freezing Status

All Phase 4 machine-readable artifacts, checkpoints, normalizers, and calibration parameters are **100% reconciled and frozen**. Phase 5 design may safely reference this single authoritative baseline.
