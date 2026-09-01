# Local Storage & Generator Architecture Inventory Report

- **Audit Scope**: Comprehensive scan of `/mnt/ai-storage/aigc_data` and `/home/manan/aigc_robust_detection`
- **Governed Training Corpus**: Manifest v6 (244,255 TRAIN images)

## 1. Generator Family Representation & Gap Analysis

| Family ID | Generator Architecture Name | Category | Manifest v6 TRAIN | Local Storage Status | Inventory Verdict | Priority Decision |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **GLIDE** | GLIDE (Guided Language-to-Image Diffusion for Generation and Editing) | Pixel-Space Cascaded Diffusion (No VAE Latents) | 0 | ABSENT | **`ABSENT`** | HIGH_PRIORITY_EXPANSION |
| **ADM** | Ablated Diffusion Models (ADM / Guided Diffusion) | Pixel-Space Guided Diffusion Models | 0 | ABSENT | **`ABSENT`** | HIGH_PRIORITY_EXPANSION |
| **BigGAN** | BigGAN / Generative Adversarial Networks | Adversarial Generative Network (Non-Diffusion) | 0 | ABSENT | **`ABSENT`** | HIGH_PRIORITY_EXPANSION |
| **VQDM** | Vector Quantized Diffusion Models (VQ-Diffusion) | Discrete Latent Codebook Diffusion | 0 | ABSENT | **`ABSENT`** | HIGH_PRIORITY_EXPANSION |
| **Wukong** | Wukong Text-to-Image Diffusion | Bilingual / Cross-Lingual Latent Diffusion | 0 | ABSENT | **`ABSENT`** | HIGH_PRIORITY_EXPANSION |
| **SD2_x** | Stable Diffusion 2.0 / 2.1 (v-prediction, 768px) | v-Objective Latent Diffusion (OpenCLIP ViT-H) | 0 | PRESENT_WEAK (Small unverified samples in diverse pool) | **`PRESENT_WEAK`** | HIGH_PRIORITY_EXPANSION |
| **DALLE2_UnCLIP** | DALL-E 2 / UnCLIP Architecture | Cascaded Diffusion + CLIP Latent Prior | 0 | ABSENT | **`PUBLIC_TRAINING_DATA_UNAVAILABLE`** | RESTRICTED (Proprietary OpenAI, research-equivalent cascaded diffusion sought) |
| **FLUX_SD3** | FLUX.1 / Stable Diffusion 3 (MMDiT / Rectified Flow) | Rectified Flow / Multimodal Diffusion Transformer (MMDiT) | 0 | PRESENT_BUT_UNUSABLE (Compressed raw archives ~62 GB, small 10-sample unpacked test folders) | **`PRESENT_BUT_UNUSABLE`** | TARGETED_LOCAL_UNPACKING |
| **SDXL_Midjourney** | SDXL & Midjourney v5/v6 | Large-Scale Latent Diffusion Ensemble | 16,390 | PRESENT_STRONG | **`PRESENT_STRONG`** | SUFFICIENT (No expansion needed) |
| **Photorealism_FineTunes** | Quality Paradox / RealisticVision Fine-Tunes | High-Fidelity Photorealism Latent Diffusion | 22,569 | PRESENT_STRONG | **`PRESENT_STRONG`** | SUFFICIENT (No expansion needed) |

## 2. Identified True Gaps Requiring Public Expansion

1. **Pixel-Space Diffusion (`GLIDE`, `ADM`)**:
   - Completely absent from current training. Because they do not use a VAE latent autoencoder, their spatial residual and frequency characteristics are distinct.
2. **Adversarial Non-Diffusion (`BigGAN`)**:
   - Completely absent. Crucial to prevent the detector from memorizing diffusion-specific denoising steps as the only definition of synthetic imagery.
3. **Discrete Latent Diffusion (`VQDM`)**:
   - Completely absent. Uses discrete codebook quantization rather than continuous Gaussian latent noise.
4. **Multilingual Diffusion (`Wukong`)**:
   - Completely absent. Evaluates cross-lingual prompt text-encoder conditioning.

## 3. Recommended Targeted Expansion Strategy

- **Primary Candidate**: `TheKernel01/Tiny-GenImage` (Hugging Face).
- **Included Novel Families**: Exactly covers `GLIDE`, `ADM`, `BigGAN`, `VQDM`, `Wukong`, plus 3,500 square `ImageNet` natural photographs.
- **Size & Bandwidth**: 28,000 samples (~8.3 GB total), perfectly within the 30 GB quota.
- **License**: Verified `CC BY-NC-SA 4.0` (Academic / Non-commercial research).
