# Final Manifest Reconciliation Report (v4)

**Generated**: 2026-08-29T06:54:44Z
**Reconciliation Status**: `FINAL_TRAINING_MANIFEST_VALID = FAILED (POPULATION_DEFICIT)`

## 1. Canonical Manifest Audit (`manifests/final_284500_governed_manifest.jsonl`)

- **Manifest File**: `/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest.jsonl`
- **Manifest SHA-256**: `130b193e6067776e957fadf3e744dfaf7a7417ba42f62ca5c171dd19dbe7eaa1`
- **Total Rows Assembled from Unpacked Pool**: `167,419`
- **OOD Contamination Rows**: `0` (Zero rows matching `Synthbuster`, `AIGIBench`, `Chameleon`, `VCT2`, `WildRF`, `SynthWildX`)

## 2. Partition Breakdown & Split Disjointness

| Partition | Total Rows | REAL | AIGC | Split Disjointness Intersections |
| :--- | :---: | :---: | :---: | :---: |
| **TRAIN** | `146,791` | `51,198` | `95,593` | `TRAIN ∩ DEV = 0` |
| **DEV** | `9,236` | `4,236` | `5,000` | `TRAIN ∩ CAL = 0` |
| **CALIBRATION** | `1,076` | `0` | `1,076` | `TRAIN ∩ TEST = 0` |
| **INTERNAL TEST** | `10,316` | `4,238` | `6,078` | `DEV ∩ CAL = 0`, `DEV ∩ TEST = 0`, `CAL ∩ TEST = 0` |

*Verification*: **All 6 pairwise set intersections are identically 0.** Disjointness is fully verified.

## 3. Governed Target vs Assembled Population

| Partition | Governed Target | Assembled | Deficit | Status |
| :--- | :---: | :---: | :---: | :---: |
| **TRAIN Total** | **`260,184`** | `146,791` | **`-113,393`** | **DEFICIT** |
| - *Train REAL* | `149,000` | `51,198` | `-97,802` | **DEFICIT** |
| - *Train AIGC* | `111,184` | `95,593` | `-15,591` | **DEFICIT** |
| **DEV** | `10,000` | `9,236` | `-764` | **DEFICIT** |
| **CALIBRATION** | `4,000` | `1,076` | `-2,924` | **DEFICIT** |
| **INTERNAL TEST** | `10,316` | `10,316` | `0` | **LOCKED & PRESERVED** |
| **Grand Total** | **`284,500`** | `167,419` | **`-117,081`** | **DEFICIT** |

## 4. Source Reserves for Parquet Manifest Extraction

The storage holds $483,084$ additional raw rows across Parquet archives to fulfill the remaining deficit without duplication:
- `parquet/HFCF_small_*.parquet` (AIGC synthetic): **$152,621$ rows**
- `defactify/data/*.parquet` (Real + AIGC pairs): **$96,000$ rows**
- `sid_parquet/train-*.parquet` (Diffusion synthetic): **$43,044$ rows**
- `aigi_quality_paradox/data/*.parquet` (Quality Paradox AIGC): **$24,000$ rows**

## 5. Operational Conclusion

`FINAL_TRAINING_MANIFEST_VALID = FAILED` because the assembled corpus provides $146,791$ training rows vs the required $260,184$. No oversampling or duplication was performed. Training remains strictly blocked pending Parquet row extraction.
