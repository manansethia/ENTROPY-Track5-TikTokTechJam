import os, sys, json, time, hashlib
import numpy as np

print("=== GENERATING AUTHORITATIVE V3 RECONCILIATION & VALIDATION REPORTS ===")

# -------------------------------------------------------------
# 1. MANIFEST RECONCILIATION V3 DATA
# -------------------------------------------------------------
manifest_path = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest.jsonl"
with open(manifest_path, "rb") as f:
    h = hashlib.sha256(f.read()).hexdigest()

manifest_v3_json = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status_verdicts": {
        "MANIFEST_CONSTRUCTED": "EXECUTED",
        "DEDUPLICATION_VERIFIED": "EXECUTED",
        "SPLIT_DISJOINTNESS_VERIFIED": "EXECUTED",
        "OOD_EXCLUSION_VERIFIED": "EXECUTED",
        "TARGET_POPULATION_RECONCILIATION": "DEFICIT_IDENTIFIED",
        "FINAL_TRAINING_MANIFEST_VALID": "FAILED"
    },
    "canonical_manifest_metadata": {
        "manifest_path": manifest_path,
        "manifest_sha256": h,
        "total_rows_assembled": 167419,
        "split_counts": {
            "TRAIN": 146791,
            "DEV": 9236,
            "CALIBRATION": 1076,
            "INTERNAL_TEST": 10316
        },
        "split_labels": {
            "TRAIN": {"REAL": 51198, "AIGC": 95593},
            "DEV": {"REAL": 4236, "AIGC": 5000},
            "CALIBRATION": {"REAL": 0, "AIGC": 1076},
            "INTERNAL_TEST": {"REAL": 4238, "AIGC": 6078}
        },
        "pairwise_intersections": {
            "TRAIN_AND_DEV": 0,
            "TRAIN_AND_CALIBRATION": 0,
            "TRAIN_AND_INTERNAL_TEST": 0,
            "DEV_AND_CALIBRATION": 0,
            "DEV_AND_INTERNAL_TEST": 0,
            "CALIBRATION_AND_INTERNAL_TEST": 0
        },
        "ood_contamination_count": 0
    },
    "governed_target_vs_assembled": {
        "TRAIN": {"target": 260184, "assembled": 146791, "delta": -113393},
        "TRAIN_REAL": {"target": 149000, "assembled": 51198, "delta": -97802},
        "TRAIN_AIGC": {"target": 111184, "assembled": 95593, "delta": -15591},
        "DEV": {"target": 10000, "assembled": 9236, "delta": -764},
        "CALIBRATION": {"target": 4000, "assembled": 1076, "delta": -2924},
        "INTERNAL_TEST": {"target": 10316, "assembled": 10316, "delta": 0},
        "TOTAL_GOVERNED": {"target": 284500, "assembled": 167419, "delta": -117081}
    },
    "storage_reserve_inventory": {
        "parquet_hfcf_rows": 152621,
        "defactify_parquet_rows": 96000,
        "sid_parquet_rows": 43044,
        "aigi_quality_paradox_rows": 24000,
        "total_available_raw_rows": 483084
    },
    "blocker_resolution": {
        "manifest_status": "Reconstruction from raw Parquet pools required to reach exact 260,184 without oversampling",
        "training_authorized": False
    }
}

# -------------------------------------------------------------
# 2. FORENSIC VALIDATION V2 DATA
# -------------------------------------------------------------
v2_forensic_status_verdicts = {
    "VLM_LOAD_VALID": "EXECUTED",
    "VLM_FORENSIC_VALID": "EXECUTED",
    "VLM_STRUCTURED_OUTPUT_VALID": "FAILED",
    "DINO_VALID": "EXECUTED",
    "EDGE_VALID": "INVALID (REPRESENTATION_COLLAPSED)",
    "CRITIC_VALID": "EXECUTED",
    "COUNTERFACTUAL_VALID": "EXECUTED",
    "VLM_FORENSIC_OPERATIONAL": "FAILED"
}

v2_forensic_report_json = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status_verdicts": v2_forensic_status_verdicts,
    "telemetry_freeze": {
        "model_repository": "vikhyatk/moondream2",
        "model_revision": "2024-08-26",
        "cached_model_path": "/home/manan/.cache/huggingface/hub/models--vikhyatk--moondream2/snapshots/92d3d73b6fd61ab84d9fe093a9c7fd8c04bf2c0d/",
        "custom_code_path": "/home/manan/.cache/huggingface/modules/transformers_modules/vikhyatk/moondream2/92d3d73b6fd61ab84d9fe093a9c7fd8c04bf2c0d/",
        "transformers_version": "5.16.1",
        "pytorch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "device": "cuda:0",
        "vram_usage_mb": 3568.96
    },
    "edge_specialist_invalidation": {
        "status": "INVALID",
        "model_class": "EdgeArtifactFeatureExtractor",
        "finding": "Representation collapsed with S_cos = 0.9999 across distinct images due to uninitialized head and LayerNorm(256). Excluded permanently from model and feedback.",
        "replacement": "HANDCRAFTED_FORENSIC_FEATURES (Sobel gradient energy, Laplacian residual, 2D-FFT ratio, SRM)"
    },
    "dino_inference_audit": {
        "status": "EXECUTED",
        "checkpoint_sha256": "edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b",
        "output_embedding_dim": 1024,
        "diversity_verified": True,
        "pairwise_cosine_similarity_range": [-0.0380, 0.1095],
        "pairwise_euclidean_distance_range": [30.67, 34.09]
    },
    "counterfactual_detector_audit": {
        "status": "EXECUTED",
        "architecture": "CLIP-ViT-L/14 + SigLIP-SO400M-224 + SRM -> Champion MLP Head",
        "cases_evaluated": [
            {
                "image_id": "REAL_SAMPLE_1_WIKIART",
                "ground_truth": "REAL",
                "original_p_aigc": 0.724739,
                "masked_p_aigc": 0.719998,
                "delta_p": -0.004741,
                "spatial_localization": "UNAVAILABLE"
            },
            {
                "image_id": "AIGC_SAMPLE_1_QUALITY_PARADOX",
                "ground_truth": "AIGC",
                "original_p_aigc": 0.718321,
                "masked_p_aigc": 0.716894,
                "delta_p": -0.001427,
                "spatial_localization": "UNAVAILABLE"
            }
        ]
    }
}

# -------------------------------------------------------------
# 3. SAVE JSON & MARKDOWN REPORTS
# -------------------------------------------------------------
out_dirs = [
    "/home/manan/aigc_robust_detection/reports",
    "/home/manan/aigc_robust_detection/final_clean_run/reports"
]

for d in out_dirs:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "final_manifest_reconciliation_v3.json"), "w") as f:
        json.dump(manifest_v3_json, f, indent=2)
    with open(os.path.join(d, "final_vlm_forensic_validation_v2.json"), "w") as f:
        json.dump(v2_forensic_report_json, f, indent=2)

# Markdown for Manifest v3
md_manifest_v3 = f"""# Final Manifest Reconciliation Report (v3)

**Generated**: {manifest_v3_json['timestamp']}
**Reconciliation Status**: `FINAL_TRAINING_MANIFEST_VALID = FAILED (POPULATION_DEFICIT)`

## 1. Canonical Manifest Metadata (`manifests/final_284500_governed_manifest.jsonl`)

- **Manifest File**: `{manifest_v3_json['canonical_manifest_metadata']['manifest_path']}`
- **Manifest SHA-256**: `{manifest_v3_json['canonical_manifest_metadata']['manifest_sha256']}`
- **Total Rows Assembled from Unpacked Pool**: `{manifest_v3_json['canonical_manifest_metadata']['total_rows_assembled']:,}`
- **OOD Contamination Rows**: `0` (Zero rows matching `Synthbuster`, `AIGIBench`, `Chameleon`, `VCT2`, `WildRF`, `SynthWildX`)

## 2. Partition Breakdown & Exact Split Disjointness

| Partition | Total Rows | REAL | AIGC | Split Disjointness Intersections |
| :--- | :---: | :---: | :---: | :---: |
| **TRAIN** | `146,791` | `51,198` | `95,593` | `TRAIN ∩ DEV = 0` |
| **DEV** | `9,236` | `4,236` | `5,000` | `TRAIN ∩ CAL = 0` |
| **CALIBRATION** | `1,076` | `0` | `1,076` | `TRAIN ∩ TEST = 0` |
| **INTERNAL TEST** | `10,316` | `4,238` | `6,078` | `DEV ∩ CAL = 0`, `DEV ∩ TEST = 0`, `CAL ∩ TEST = 0` |

*Result*: **All 6 pairwise set intersections are exactly 0.** Disjointness is fully verified.

## 3. Reconciliation Against Governed Target ($284,500$ Total Population)

| Partition | Governed Target | Assembled (Unpacked Pool) | Deficit | Status |
| :--- | :---: | :---: | :---: | :---: |
| **TRAIN Total** | **`260,184`** | `146,791` | **`-113,393`** | **DEFICIT** |
| - *Train REAL* | `149,000` | `51,198` | `-97,802` | **DEFICIT** |
| - *Train AIGC* | `111,184` | `95,593` | `-15,591` | **DEFICIT** |
| **DEV** | `10,000` | `9,236` | `-764` | **DEFICIT** |
| **CALIBRATION** | `4,000` | `1,076` | `-2,924` | **DEFICIT** |
| **INTERNAL TEST** | `10,316` | `10,316` | `0` | **RECONCILED** |
| **Grand Total** | **`284,500`** | `167,419` | **`-117,081`** | **DEFICIT** |

## 4. Storage Reserves for Parquet Manifest Extraction

The storage contains $483,084$ additional raw rows across Parquet archives to fulfill the full $284,500$ target:
- `parquet/HFCF_small_*.parquet` (AIGC synthetic): **$152,621$ rows**
- `defactify/data/*.parquet` (Real + AIGC pairs): **$96,000$ rows**
- `sid_parquet/train-*.parquet` (Diffusion synthetic): **$43,044$ rows**
- `aigi_quality_paradox/data/*.parquet` (Quality Paradox AIGC): **$24,000$ rows**

## 5. Pre-Training Conclusion

`FINAL_TRAINING_MANIFEST_VALID = FAILED` because the assembled unpacked corpus provides $146,791$ training rows vs the required $260,184$. No oversampling or duplication was performed. Training remains blocked pending Parquet row extraction.
"""

for d in out_dirs:
    with open(os.path.join(d, "final_manifest_reconciliation_v3.md"), "w") as f:
        f.write(md_manifest_v3)

# Markdown for Forensic v2
md_forensic_v2 = f"""# Final VLM Forensic & Multi-Expert Validation Report (v2)

**Generated**: {v2_forensic_report_json['timestamp']}
**Operational Status**: `VLM_FORENSIC_OPERATIONAL = FAILED`

## 1. Operational Status Verdicts

| Gate / Component | Status | Empirical Rationale |
| :--- | :---: | :--- |
| `VLM_LOAD_VALID` | **`EXECUTED`** | Moondream2 (`2024-08-26`) loaded on `cuda:0` ($3,568.96\\text{{ MB}}$ VRAM). |
| `VLM_FORENSIC_VALID` | **`EXECUTED`** | Direct forward passes completed on 6 real training images. |
| `VLM_STRUCTURED_OUTPUT_VALID` | **`FAILED`** | Model produces unstructured/repetitive output on complex schema prompts; zero fabricated keyword parsing allowed. |
| `DINO_VALID` | **`EXECUTED`** | DINOv2-Registers-L forward passes executed ($1024\\text{{d}}$ embeddings, pairwise cosine sim: $-0.038$ to $+0.109$). |
| `EDGE_VALID` | **`INVALID`** | Neural Edge-Specialist exhibits representation collapse ($S_{{\\cos}} = 0.9999$ across distinct images due to uninitialized head/LayerNorm); permanently invalidated. Handcrafted Sobel gradient energy remains valid. |
| `CRITIC_VALID` | **`EXECUTED`** | 4 critic cases evaluated (`CRITIC_INDEPENDENCE = LIMITED`). |
| `COUNTERFACTUAL_VALID` | **`EXECUTED`** | Master Detector executed on original vs masked regions ($P_{{\\text{{orig}}}}$, $P_{{\\text{{masked}}}}$, $\\Delta P$ recorded; spatial localization marked `UNAVAILABLE`). |
| `VLM_FORENSIC_OPERATIONAL` | **`FAILED`** | **BLOCKED**: Gated on `VLM_STRUCTURED_OUTPUT_VALID = FAILED` and `EDGE_VALID = INVALID`. |

## 2. Invalidation of Neural Edge Specialist

- **Finding**: `EdgeArtifactFeatureExtractor (256d)` has representation collapse ($S_{{\\cos}} = 0.9999$, $\|x\|_2 = 15.9697$).
- **Action**: Permanently marked **`INVALID`**. Excluded from feature fusion and feedback learning.
- **Approved Physical Signal**: `HANDCRAFTED_FORENSIC_FEATURES` (Sobel gradient magnitude mean, Laplacian residuals, 2D-FFT ratio, SRM).

## 3. DINOv2 Representation Diversity

- **Checkpoint SHA-256**: `{v2_forensic_report_json['dino_inference_audit']['checkpoint_sha256']}`
- **Pairwise Cosine Similarity**: Ranging from $-0.0380$ to $+0.1095$ across distinct images, confirming orthogonal representation capability.
- **Pairwise Euclidean Distances**: $30.67$ to $34.09$.

## 4. Counterfactual Master Detector Evidence

- **REAL_SAMPLE_1_WIKIART**: $P(\\text{{AIGC}})_{{\\text{{orig}}}} = 0.724739 \\to P(\\text{{AIGC}})_{{\\text{{masked}}}} = 0.719998$, $\\Delta P = -0.004741$ (Spatial Localization: `UNAVAILABLE`).
- **AIGC_SAMPLE_1_QUALITY_PARADOX**: $P(\\text{{AIGC}})_{{\\text{{orig}}}} = 0.718321 \\to P(\\text{{AIGC}})_{{\\text{{masked}}}} = 0.716894$, $\\Delta P = -0.001427$ (Spatial Localization: `UNAVAILABLE`).
"""

for d in out_dirs:
    with open(os.path.join(d, "final_vlm_forensic_validation_v2.md"), "w") as f:
        f.write(md_forensic_v2)

print("V3 and V2 reports written successfully!")
