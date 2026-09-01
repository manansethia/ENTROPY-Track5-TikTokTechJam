# Phase 3 Step 3 Data Integrity & Provenance Reconciliation Report

*Audit Timestamp*: `2026-08-29T07:22:33Z`
*Manifest File*: `/home/manan/aigc_robust_detection/manifests/phase2_150k_manifest.jsonl` (SHA-256: `91bcd1de69689017859fa275825bed146aaf241ef71e57eb64f5562c615ceb23`)
*Audit Verdict*: **`PASSED — 100% CRYPTOGRAPHIC ISOLATION & ZERO CONTAMINATION`**

## 1. Validation Set Composition (`PHASE2_VAL`, N=10,312)

- **Total Samples**: **`10,312`** ($4,236$ Real [41.08%] / $6,076$ AIGC [58.92%])
- **Source Datasets**:
  - `loose_synthetic_corpus`: 3632 images (35.2%)
  - `wikiart_fine_art`: 2453 images (23.8%)
  - `aigi_quality_paradox`: 2444 images (23.7%)
  - `loose_authentic_corpus`: 1783 images (17.3%)
- **Generator Families**:
  - `Synthetic_HighFrequency_CF`: 2982 images (28.9%)
  - `Authentic_WikiArt_FineArt`: 2453 images (23.8%)
  - `Synthetic_QualityParadox_ModernDiffusion`: 2444 images (23.7%)
  - `Authentic_Real_General`: 1472 images (14.3%)
  - `Synthetic_SID_Diffusion`: 650 images (6.3%)
  - `Authentic_COCO`: 311 images (3.0%)

## 2. Probe Training Subset Composition (`PHASE2_TRAIN`, N=20,000)

- **Total Samples**: **`20,000`** ($8,220$ Real [40.79%] / $11,780$ AIGC [59.21%])
- **Source Datasets**:
  - `loose_synthetic_corpus`: 7138 images (35.7%)
  - `wikiart_fine_art`: 4904 images (24.5%)
  - `aigi_quality_paradox`: 4704 images (23.5%)
  - `loose_authentic_corpus`: 3254 images (16.3%)
- **Generator Families**:
  - `Synthetic_HighFrequency_CF`: 5824 images (29.1%)
  - `Authentic_WikiArt_FineArt`: 4904 images (24.5%)
  - `Synthetic_QualityParadox_ModernDiffusion`: 4704 images (23.5%)
  - `Authentic_Real_General`: 2695 images (13.5%)
  - `Synthetic_SID_Diffusion`: 1314 images (6.6%)
  - `Authentic_COCO`: 559 images (2.8%)

## 3. Cryptographic Partition Isolation & Leakage Verification

| Integrity Check | Overlap Metric | Status | Evidence |
| :--- | :--- | :--- | :--- |
| `probe_train ∩ validation` | 0 hashes / 0 paths | **PASSED** | Zero sample overlap |
| `probe_train ∩ internal_test` | 0 hashes | **PASSED** | Zero sample overlap |
| `validation ∩ internal_test` | 0 hashes | **PASSED** | Zero sample overlap |
| `full_train ∩ internal_test` | 0 hashes | **PASSED** | Zero sample overlap |
| `Quarantined OOD Contamination` | 0 samples | **PASSED** | Synthbuster & AIGIBench 100% Isolated |

## 4. Methodological Verification

- **Normalization Source**: Strictly fitted on probe-training split only; validation set normalized using frozen training statistics.
- **Probe Fitting Source**: Probes fitted exclusively on 20,000 probe-training samples.
- **Validation Evaluation Role**: Validation set used strictly for evaluation, correlation, error forensics, and complementarity analysis.
- **Calibration Role**: Calibration evaluated on dedicated validation sub-partition (no test set leakage).
- **Internal Test Guardrail**: Internal test remains 100% isolated until final single-run frozen comparison.
- **External Ood Guardrail**: Synthbuster (9,000 images) and AIGIBench remain strictly locked until post-training evaluation.
