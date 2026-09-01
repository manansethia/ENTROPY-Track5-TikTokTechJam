# OOD Remediation Data Policy & Sampling Strategy

- **Governed Remediation Manifest**: `manifests/ood_remediation_manifest_v1.jsonl`
- **Total Ingested Expansion Samples**: `21,000` novel images (3,500 per target category)

## 1. Remediation Sampling & Balancing Policy

1. **Class Balance**: 50% REAL, 50% AIGC in every training batch.
2. **Generator Uniformity**: Uniform sampling across all 12 AIGC generator families (`GLIDE`, `ADM`, `BigGAN`, `VQDM`, `Wukong`, `SDXL/MJ`, `SID`, `Quality Paradox`, `Diverse`, `Diffusion Synthetics`, `Defactify`, `Latent Diffusion`).
3. **Geometric & Aspect-Ratio Invariance**: Active augmentation pipeline breaks the 512x512 square shortcut identified in Stage 1.
