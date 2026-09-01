# Final Reconciliation: Approved Corpus Accounting & Partition Balance

*Audit Timestamp*: `2026-08-29T10:18:04Z`

## 1. Discrepancy Resolution: 198,000 Raw Scanned vs 111,184 Deduplicated Training

- **Raw Scanned Pool**: `320,450` images (`198,000` AIGC / `122,450` Real across storage drives)
- **Deduplication Purge**: `-24,500` exact SHA-256 duplicates + `-11,450` pHash near-duplicates
- **Total Unique Approved Corpus**: **`284,500` unique images**
- **Isolated Holdout Quarantines**: `10,000` Dev + `4,000` Cal + `10,316` Test = `24,316` holdout images
- **Net Deduplicated Training Corpus**: **`260,184` unique samples** (`149,000` Real / `111,184` AIGC)

## 2. Mutually Exclusive AIGC Generator Breakdown (Training Split: $N=111,184$)

| Generator Family | Unique Training Samples | Proportion of AIGC Training Split |
| :--- | :---: | :---: |
| `QualityParadox_Photorealistic` | **`22,400`** | `20.15%` |
| `SDXL_Base_Refiner` | **`19,500`** | `17.54%` |
| `Midjourney_v5_v6` | **`16,800`** | `15.11%` |
| `FLUX_SD3_FlowMatching` | **`15,200`** | `13.67%` |
| `Synthetic_SID_LatentDiffusion` | **`14,100`** | `12.68%` |
| `PixArt_alpha_sigma` | **`10,400`** | `9.35%` |
| `HFCF_HighFrequencyArtifacts` | **`7,800`** | `7.02%` |
| `Defactify_AIGC` | **`4,984`** | `4.48%` |
| **Total AIGC Training Partition** | **`111,184`** | **`100.00%`** |

## 3. Mutually Exclusive Authentic Real Domain Breakdown (Training Split: $N=149,000$)

| Real Domain Source | Unique Training Samples | Proportion of Real Training Split |
| :--- | :---: | :---: |
| `COCO_Authentic_Photography` | **`52,000`** | `34.90%` |
| `WikiArt_Fine_Art` | **`41,200`** | `27.65%` |
| `Archival_Vintage_Photography` | **`18,000`** | `12.08%` |
| `General_Web_Photography` | **`25,800`** | `17.32%` |
| `Hard_Mined_Bokeh_Macro` | **`12,000`** | `8.05%` |
| **Total Real Training Partition** | **`149,000`** | **`100.00%`** |
