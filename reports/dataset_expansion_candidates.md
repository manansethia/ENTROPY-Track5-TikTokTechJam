# Dataset Expansion Candidates & Provenance Audit

- **Candidate Source**: [`TheKernel01/Tiny-GenImage`](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)
- **License**: `CC BY-NC-SA 4.0` (Attribution-NonCommercial-ShareAlike 4.0 International)
- **License Verified**: `YES` (Date Checked: 2026-08-30)
- **Total Newly Added Samples**: **`13,500`**
- **Quarantined Hash Duplicates**: `0`
- **New Remediation Manifest**: `/home/manan/aigc_robust_detection/manifests/ood_remediation_manifest_v1.jsonl`

## 1. Targeted Generator Architecture Ingestion Matrix

| Generator Family | Architecture Category | Class | Samples Added | License | Status | Scientific Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ImageNet_Authentic_Photo** | REAL_PHOTOGRAPHY | REAL | `3,500` | CC BY-NC-SA 4.0 | **`INCORPORATED`** | Targets novel architectural representation for REAL_PHOTOGRAPHY. |
| **ADM_PixelDiffusion** | Pixel_Space_Guided_Diffusion | AIGC | `2,000` | CC BY-NC-SA 4.0 | **`INCORPORATED`** | Targets novel architectural representation for Pixel_Space_Guided_Diffusion. |
| **BigGAN_Adversarial** | Generative_Adversarial_Network | AIGC | `2,000` | CC BY-NC-SA 4.0 | **`INCORPORATED`** | Targets novel architectural representation for Generative_Adversarial_Network. |
| **GLIDE_PixelDiffusion** | Text_Guided_Pixel_Diffusion | AIGC | `2,000` | CC BY-NC-SA 4.0 | **`INCORPORATED`** | Targets novel architectural representation for Text_Guided_Pixel_Diffusion. |
| **Midjourney_v5** | Latent_Diffusion_Ensemble | AIGC | `0` | CC BY-NC-SA 4.0 | **`EXCLUDED (Redundant with existing v6)`** | Already strongly represented in base corpus. |
| **SD14_LatentDiffusion** | Latent_Diffusion_SD14 | AIGC | `0` | CC BY-NC-SA 4.0 | **`EXCLUDED (Redundant with existing v6)`** | Already strongly represented in base corpus. |
| **SD15_LatentDiffusion** | Latent_Diffusion_SD15 | AIGC | `0` | CC BY-NC-SA 4.0 | **`EXCLUDED (Redundant with existing v6)`** | Already strongly represented in base corpus. |
| **VQDM_DiscreteDiffusion** | Discrete_Latent_Codebook_Diffusion | AIGC | `2,000` | CC BY-NC-SA 4.0 | **`INCORPORATED`** | Targets novel architectural representation for Discrete_Latent_Codebook_Diffusion. |
| **Wukong_BilingualDiffusion** | Multilingual_Latent_Diffusion | AIGC | `2,000` | CC BY-NC-SA 4.0 | **`INCORPORATED`** | Targets novel architectural representation for Multilingual_Latent_Diffusion. |

## 2. Leakage & Overlap Protection Verification

1. **Zero Overlap with Locked Benchmarks**: Exact SHA-256 checksums verified against `INTERNAL_TEST`, `Synthbuster`, and `AIGIBench`.
2. **Zero Split Contamination**: All new samples are injected strictly into `TRAIN`. The immutable `DEV` ($10,000$ samples) and `CAL` ($4,000$ samples) splits remain completely untouched.
3. **Redundant Families Excluded**: Redundant SD 1.4, SD 1.5, and Midjourney samples from Tiny-GenImage were actively filtered out to prevent diluting the novel architectures.
