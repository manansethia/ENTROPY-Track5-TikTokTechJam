# Training-Data Coverage & Generative Architecture Audit

- **Governed TRAIN Partition**: 244,255 images (132,102 REAL, 112,153 AIGC)
- **Governed Split**: Immutable Manifest v6

## 1. Approved Training Corpus Inventory by Architecture & Domain

| Domain / Source | Class | Sample Count | Class % | Total % | Architecture Category | Resolution Profile |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **WikiArt_Fine_Art** | REAL | 77,338 | 58.54% | 31.66% | REAL_PHOTOGRAPHY_AND_ART | High-Res Varying (>1500x1000) |
| **Diverse_Generators** | AIGC | 36,227 | 32.3% | 14.83% | Pixel & Latent Diffusion (Mixed) | 512x512 |
| **COCO_Authentic_Photography** | REAL | 34,517 | 26.13% | 14.13% | REAL_PHOTOGRAPHY_AND_ART | 640x480 / 1024x683 |
| **Quality_Paradox_Photorealism** | AIGC | 22,569 | 20.12% | 9.24% | Latent Diffusion (RealisticVision / Photorealism) | 1024x1024 / 944x624 |
| **SDXL_Midjourney** | AIGC | 16,390 | 14.61% | 6.71% | Latent Diffusion (Ensemble Backbones) | 512x512 / 1024x1024 |
| **Diffusion_Synthetics** | AIGC | 15,335 | 13.67% | 6.28% | Pixel & Latent UNet Diffusion | 512x512 |
| **SID_LatentDiffusion** | AIGC | 14,112 | 12.58% | 5.78% | Latent Diffusion Models (LDM / SD 1.x) | 1024x1024 |
| **Natural_SID_Photography** | REAL | 13,689 | 10.36% | 5.6% | REAL_PHOTOGRAPHY_AND_ART | 1024x768 / 1024x680 |
| **Natural_Photography** | REAL | 6,558 | 4.96% | 2.68% | REAL_PHOTOGRAPHY_AND_ART | 1024x768 |
| **Defactify_AIGC** | AIGC | 4,679 | 4.17% | 1.92% | Cross-Generator Text-to-Image | 1024x1024 |
| **Latent_Diffusion** | AIGC | 2,841 | 2.53% | 1.16% | Latent Diffusion Models (SD 1.4) | 512x512 |

## 2. Identified Architectural Blindspots & OOD Vulnerabilities

| Generator Family | Architectural Type | TRAIN Status | Observed OOD Detection Rate | Failure Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **DALL-E 2 / UnCLIP Architecture** | Two-Stage Cascaded Diffusion + CLIP Prior | `ABSENT / HEAVILY UNDERREPRESENTED` | **`20.0% Detection Rate on Synthbuster`** | Uses unCLIP prior space + cascaded upsamplers rather than standard single-stage latent diffusion, producing distinct non-square spectral footprints. |
| **Adobe Firefly / Proprietary Commercial Diffusion** | Proprietary Commercial Diffusion with Heavy In-Line Post-Processing | `ABSENT` | **`56.0% Detection Rate on Synthbuster`** | Commercial safety filtering, aggressive color post-processing, and custom rendering pipelines smooth out high-frequency residual anomalies. |
| **Stable Diffusion 2.x (v-prediction / 768px)** | v-Objective Latent Diffusion (OpenCLIP ViT-H) | `UNDERREPRESENTED (<2%)` | **`57.0% Detection Rate on Synthbuster`** | v-prediction formulation and OpenCLIP text encoder create different latent noise dynamics compared to SD 1.x / SDXL. |
| **GLIDE / Pure Pixel-Space Diffusion** | Text-Guided Cascaded Pixel Diffusion | `ABSENT / MINIMAL` | **`77.0% Detection Rate on Synthbuster`** | Operates directly in pixel space without VAE autoencoder latent compression artifacts. |
| **Autoregressive / MaskGIT / Flow-Matching (e.g. Flux, Muse)** | Non-Diffusion (Rectified Flow, Masked Token Modeling) | `ABSENT` | **`UNKNOWN (Emerging)`** | Completely lacks diffusion denoising steps; generates tokens or straight-line ODE trajectories. |

## 3. Remediation Recommendations

1. **Data Re-Balancing**: Currently, `WikiArt` accounts for **58.5%** of all real training data, while `Diverse_Generators` accounts for **32.3%** of all AIGC data. Batches must be balanced uniformly across all 7 AIGC generator families and all 4 Real domains.
2. **Augmentation-Driven Generalization**: Since external generators (DALL-E 2, Firefly, SD 2.x) employ varied post-processing, upsampling, and non-standard compression pipelines, invariant augmentations (JPEG sweeps, bilinear downscaling, blur/sharpen, color perturbation) must be applied during training to force the model to learn deep structural anomalies rather than specific VAE or patch signatures.
