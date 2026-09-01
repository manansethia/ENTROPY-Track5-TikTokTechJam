# Final Manifest Reconciliation Report (v2)

**Generated**: 2026-08-29T06:47:15Z
**Reconciliation Status**: `FINAL_TRAINING_MANIFEST_VALID = FAILED (POPULATION_DEFICIT)`

## 1. Split Audit of Current Active Manifest (`phase2_150k_manifest.jsonl`)

- **Manifest Path**: `/home/manan/aigc_robust_detection/manifests/phase2_150k_manifest.jsonl`
- **Manifest SHA256**: `91bcd1de69689017859fa275825bed146aaf241ef71e57eb64f5562c615ceb23`
- **Total Rows Computed**: `103137`
- **PHASE2_TRAIN**: `82509` rows ({'REAL': 33895, 'AIGC': 48614})
- **PHASE2_VAL**: `10312` rows ({'REAL': 4236, 'AIGC': 6076})
- **PHASE2_INTERNAL_TEST**: `10316` rows ({'REAL': 4238, 'AIGC': 6078})
- **Pairwise Intersections**: `{'PHASE2_TRAIN_AND_PHASE2_VAL': 0, 'PHASE2_TRAIN_AND_PHASE2_INTERNAL_TEST': 0, 'PHASE2_VAL_AND_PHASE2_INTERNAL_TEST': 0}` (All **0**)
- **OOD Contamination Rows**: `0` (**0**)

## 2. Governed Target vs Active Population Comparison

| Partition | Governed Target | Active Computed | Deficit | Status |
| :--- | :---: | :---: | :---: | :---: |
| **TRAIN (Total)** | **`260,184`** | `82,509` | **`-177,675`** | **DEFICIT** |
| - *Train REAL* | `149,000` | `33,895` | `-115,105` | **DEFICIT** |
| - *Train AIGC* | `111,184` | `48,614` | `-62,570` | **DEFICIT** |
| **DEV** | `10,000` | `10,312` | `+312` | Valid |
| **CALIBRATION** | `4,000` | `0` (unassigned) | `-4,000` | **DEFICIT** |
| **INTERNAL TEST** | `10,316` | `10,316` | `0` | **RECONCILED** |
| **Total Governed** | **`284,500`** | `103,137` | **`-181,363`** | **DEFICIT** |

## 3. Storage Reserves for Manifest Assembly

The underlying storage contains sufficient raw reserves ($482,000+$ rows) to reconstruct the full governed $260,184$ training population:
- `parquet` (`HFCF` synthetic): $152,621$ rows
- `defactify` (real + synthetic pairs): $96,000$ rows
- `sid_parquet` (synthetic diffusion): $43,044$ rows
- `aigi_quality_paradox`: $24,000$ rows
- `unpacked images`: $177,242$ files

## 4. Operational Conclusion

`FINAL_TRAINING_MANIFEST_VALID = FAILED` because the current active manifest contains $82,509$ training rows rather than the required $260,184$. No oversampling or duplication has been performed. Training remains blocked pending governed manifest reconstruction.
