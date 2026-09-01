# High-Resolution Candidate Datasets Inspection & Governance Report

## 1. Executive Dataset Governance Matrix
| Dataset Name | HuggingFace Repository | Designated Role | Total Files | Downloads | Governance Action |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`NTIRE 2026 Robust`** | `deepfakesMSU/NTIRE-RobustAIGenDetection-train` | **TRAIN** | ~277K images | 2030 | **Approved for Training Pool** |
| **`HiRes-50K`** | `Mu437/HiRes-50K` | **EVALUATION ONLY** | 50,568 images | 60 | **Strictly Locked Evaluation Benchmark** |
| **`AIGC Benchmark`** | `TheKernel01/AIGC-Detection-Benchmark` | **BENCHMARK / TRAIN** | 60 parquet splits | 1322 | **Sampled for Generator Diversity** |
| **`Quality Paradox`** | `Coxy7/AIGI-Detection-Quality-Paradox` | **HARD AIGC TRAIN** | 15 parquet splits | 321 | **Approved for Hard AIGC Weighting** |
| **`MLLM Generated`** | `zr-zhang/MLLM-Generated-Image-Detection-Dataset` | **2026 FRONTIER EVAL** | 4,358 images | 2995 | **Benchmark for MLLM / GPT-Image2** |

---

## 2. Dataset Specific Profiles & Verification
### NTIRE_2026_Robust_Train (`deepfakesMSU/NTIRE-RobustAIGenDetection-train`)
- **Designated Role**: `TRAIN`
- **Description**: NTIRE 2026 Robust AI-Generated Image Detection training pool (277K images, cropping/compression robust).
- **File Structure**: .gitattributes, README.md, assets/header_NTIRE.jpg, shard_0.zip, shard_1.zip
### HiRes_50K_Benchmark (`Mu437/HiRes-50K`)
- **Designated Role**: `EVALUATION_ONLY`
- **Description**: 50,568 images from <1K to >10K resolution, up to 64MP. Strictly for evaluation.
- **File Structure**: .gitattributes, README.md, W_0900.zip, W_1200.zip, W_1500.zip
### AIGC_Detection_Benchmark (`TheKernel01/AIGC-Detection-Benchmark`)
- **Designated Role**: `TRAIN_OR_EVAL`
- **Description**: Multi-generator benchmark (DALL-E 2, Midjourney, ADM, BigGAN, StyleGAN, ProGAN).
- **File Structure**: .gitattributes, README.md, data/test-00000-of-00060.parquet, data/test-00001-of-00060.parquet, data/test-00002-of-00060.parquet
### AIGI_Quality_Paradox (`Coxy7/AIGI-Detection-Quality-Paradox`)
- **Designated Role**: `HARD_AIGC_TRAIN`
- **Description**: 24K realistic AIGC images focused on high-quality generator realism.
- **File Structure**: .gitattributes, README.md, data/fake-00000-of-00015.parquet, data/fake-00001-of-00015.parquet, data/fake-00002-of-00015.parquet
### MLLM_Generated_Dataset (`zr-zhang/MLLM-Generated-Image-Detection-Dataset`)
- **Designated Role**: `MLLM_EVAL_TRAIN`
- **Description**: 2026 benchmark for GPT Image2 and Nano Banana2 texture/structure/hybrid cases.
- **File Structure**: .gitattributes, README.md, images/Preprocessing/Hybrid Images/GPT-Image2-fake/1.jpg, images/Preprocessing/Hybrid Images/GPT-Image2-fake/10.jpg, images/Preprocessing/Hybrid Images/GPT-Image2-fake/100.jpg
