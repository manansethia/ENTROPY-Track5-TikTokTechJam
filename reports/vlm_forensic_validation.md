# Corrected VLM Forensic & Multi-Expert Validation Report

**Generated**: 2026-08-29T06:38:11Z

## 1. Operational Status Verdicts

| Component / Gate | Status |
| :--- | :---: |
| `MANIFEST_VALID` | **`EXECUTED`** |
| `OOD_EXCLUSION_VALID` | **`EXECUTED`** |
| `FOUNDATION_MODELS_VALID` | **`EXECUTED`** |
| `VLM_LOAD_VALID` | **`EXECUTED`** |
| `VLM_FORENSIC_VALID` | **`EXECUTED`** |
| `VLM_STRUCTURED_OUTPUT_VALID` | **`FAILED`** |
| `CRITIC_VALID` | **`EXECUTED`** |
| `COUNTERFACTUAL_VALID` | **`EXECUTED`** |
| `VLM_FORENSIC_OPERATIONAL` | **`FAILED`** |

## 2. Frozen Telemetry & Checkpoint Hashes

- **VLM Model**: `vikhyatk/moondream2` (`2024-08-26`)
- **DINOv2-Registers-L SHA256**: `edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b`
- **Edge-Specialist Model**: `EdgeArtifactFeatureExtractor (256d)`
- **PyTorch / Transformers / CUDA**: `2.13.0+cu130` / `5.16.1` / `cu13.0`
- **Device / VRAM**: `cuda:0` / `3568.96 MB`

## 3. Computed Manifest Disjointness

- **Manifest SHA256**: `91bcd1de69689017859fa275825bed146aaf241ef71e57eb64f5562c615ceb23`
- **Total Rows Computed**: `103137`
- **PHASE2_TRAIN**: `82509` rows ({'REAL': 33895, 'AIGC': 48614})
- **PHASE2_VAL**: `10312` rows ({'REAL': 4236, 'AIGC': 6076})
- **PHASE2_INTERNAL_TEST**: `10316` rows ({'REAL': 4238, 'AIGC': 6078})
- **Pairwise Intersections**: `{'PHASE2_TRAIN_AND_PHASE2_VAL': 0, 'PHASE2_TRAIN_AND_PHASE2_INTERNAL_TEST': 0, 'PHASE2_VAL_AND_PHASE2_INTERNAL_TEST': 0}`
- **OOD Contamination Rows**: `0`

## 4. Actual DINOv2 Inference Evidence

| Image ID | Input Shape | Output Dim | Embedding Mean | Embedding Std | L2 Norm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `REAL_SAMPLE_1_WIKIART` | `[1, 3, 224, 224]` | `1024` | `0.0117` | `0.7109` | `22.7344` |
| `REAL_SAMPLE_2_COCO_PHOTO` | `[1, 3, 224, 224]` | `1024` | `0.0035` | `0.6885` | `22.0312` |
| `REAL_SAMPLE_3_MACRO_PHOTO` | `[1, 3, 224, 224]` | `1024` | `-0.0024` | `0.7617` | `24.3594` |
| `AIGC_SAMPLE_1_QUALITY_PARADOX` | `[1, 3, 224, 224]` | `1024` | `-0.0028` | `0.7373` | `23.5781` |
| `AIGC_SAMPLE_2_CF_SYNTHETIC` | `[1, 3, 224, 224]` | `1024` | `0.0148` | `0.7266` | `23.2344` |
| `AIGC_SAMPLE_3_HFCF_SYNTHETIC` | `[1, 3, 224, 224]` | `1024` | `0.0148` | `0.7266` | `23.2344` |

## 5. Actual Edge-Specialist Inference Evidence

| Image ID | Model Class | Output Dim | Feature Mean | Feature Std | L2 Norm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `REAL_SAMPLE_1_WIKIART` | `EdgeArtifactFeatureExtractor` | `256` | `0.0` | `1.0001` | `15.9697` |
| `REAL_SAMPLE_2_COCO_PHOTO` | `EdgeArtifactFeatureExtractor` | `256` | `-0.0` | `1.0001` | `15.9697` |
| `REAL_SAMPLE_3_MACRO_PHOTO` | `EdgeArtifactFeatureExtractor` | `256` | `0.0` | `1.0001` | `15.9697` |
| `AIGC_SAMPLE_1_QUALITY_PARADOX` | `EdgeArtifactFeatureExtractor` | `256` | `0.0` | `1.0001` | `15.9697` |
| `AIGC_SAMPLE_2_CF_SYNTHETIC` | `EdgeArtifactFeatureExtractor` | `256` | `-0.0` | `1.0001` | `15.9697` |
| `AIGC_SAMPLE_3_HFCF_SYNTHETIC` | `EdgeArtifactFeatureExtractor` | `256` | `-0.0` | `1.0001` | `15.9697` |

## 6. Counterfactual Detector Test (Actual Detector Inference)

### REAL_SAMPLE_1_WIKIART (REAL)
- **VLM Claimed Region**: `UNAVAILABLE`
- **Spatial Localization Status**: `SPATIAL_LOCALIZATION_UNAVAILABLE`
- **Masked Bounding Box (Pixels)**: `[524.2, 345.5, 1572.8, 1036.5]`
- **Original Detector $P(\text{AIGC})$**: `0.724739`
- **Masked Detector $P(\text{AIGC})$**: `0.719998`
- **$\Delta P(\text{AIGC})$**: `-0.004741`

### AIGC_SAMPLE_1_QUALITY_PARADOX (AIGC)
- **VLM Claimed Region**: `UNAVAILABLE`
- **Spatial Localization Status**: `SPATIAL_LOCALIZATION_UNAVAILABLE`
- **Masked Bounding Box (Pixels)**: `[156.0, 236.0, 468.0, 708.0]`
- **Original Detector $P(\text{AIGC})$**: `0.718321`
- **Masked Detector $P(\text{AIGC})$**: `0.716894`
- **$\Delta P(\text{AIGC})$**: `-0.001427`

