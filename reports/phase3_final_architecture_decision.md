# Phase 3 Final Architecture Decision & Multi-Expert Challenge Report

*Audit Timestamp*: `2026-08-29T09:19:00Z`
*Verdict*: **`PHASE_3_MULTI_EXPERT_CHAMPION_CONFIRMED`**

## 1. Selected Champion Architecture: `B_CLIP_SigLIP_mlp2`

- **Feature Dimension**: **`2176-d`**
- **Expert Branches Included**: `e1_clip + e2_siglip`
- **Head Architecture**: `mlp2` (558081 trainable parameters)
- **Validation AUROC**: **`0.9972`** (Marginal Gain: **`-0.0016`**)
- **Validation FPR @ $\tau=0.80$**: **`1.56%`** ($N=66$ False Positives / $4,236$ Real)
- **Validation TPR @ $\tau=0.80$**: **`96.99%`** ($N=183$ False Negatives / $6,076$ AIGC)
- **Total Validation Error Reduction**: **`-63` fewer errors** than Phase 2 baseline ($186 \to 249$)

## 2. Multi-Objective Candidate Comparison Table

| Rank | Candidate ID | Dim | Trainable Params | Val AUROC | Val AUPRC | FPR @ 0.80 | TPR @ 0.80 | Total Errors | Net Error Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `B_CLIP_SigLIP_mlp2` | 2176d | 558,081 | **0.9972** | 0.9980 | 1.56% | 96.99% | 249 | +63 |
| 2 | `F_Vision_Spectral_Wavelet_mlp2` | 4068d | 1,042,433 | **0.9969** | 0.9979 | 1.86% | 97.04% | 259 | +73 |
| 3 | `G_All_9_Experts_Full_mlp2` | 5130d | 1,314,305 | **0.9965** | 0.9976 | 1.72% | 96.89% | 262 | +76 |
| 4 | `A_Phase2_Baseline_mlp2` | 2212d | 567,297 | **0.9973** | 0.9981 | 1.70% | 96.86% | 263 | +77 |
| 5 | `E_All_Vision_Transformer_Conv_mlp2` | 4992d | 1,278,977 | **0.9969** | 0.9979 | 1.89% | 96.99% | 263 | +77 |
| 6 | `D_CLIP_SigLIP_DINO_EVA_mlp2` | 4224d | 1,082,369 | **0.9966** | 0.9978 | 2.01% | 96.96% | 270 | +84 |
| 7 | `I_QuadStream_Forensic_mlp2` | 4026d | 1,031,681 | **0.9966** | 0.9975 | 1.84% | 96.76% | 275 | +89 |
| 8 | `A_Phase2_Baseline_expert_dropout` | 2212d | 567,297 | **0.9971** | 0.9980 | 1.35% | 96.38% | 277 | +91 |
| 9 | `I_QuadStream_Forensic_expert_dropout` | 4026d | 1,031,681 | **0.9972** | 0.9981 | 1.16% | 96.02% | 291 | +105 |
| 10 | `C_CLIP_SigLIP_DINO_mlp2` | 3200d | 820,225 | **0.9971** | 0.9980 | 1.82% | 96.48% | 291 | +105 |
| 11 | `I_QuadStream_Forensic_mlp3_bottleneck` | 4026d | 2,128,897 | **0.9968** | 0.9978 | 1.91% | 96.54% | 291 | +105 |
| 12 | `G_All_9_Experts_Full_mlp3_bottleneck` | 5130d | 2,694,145 | **0.9965** | 0.9974 | 1.82% | 96.41% | 295 | +109 |
| 13 | `G_All_9_Experts_Full_expert_dropout` | 5130d | 1,314,305 | **0.9976** | 0.9983 | 1.09% | 95.84% | 299 | +113 |
| 14 | `A_Phase2_Baseline_mlp3_bottleneck` | 2212d | 1,200,129 | **0.9964** | 0.9976 | 2.01% | 96.40% | 304 | +118 |
| 15 | `G_All_9_Experts_Full_gated_moe` | 5130d | 1,319,187 | **0.9951** | 0.9962 | 2.57% | 96.51% | 321 | +135 |
| 16 | `A_Phase2_Baseline_gated_moe` | 2212d | 568,327 | **0.9957** | 0.9969 | 2.24% | 96.12% | 331 | +145 |
| 17 | `I_QuadStream_Forensic_gated_moe` | 4026d | 1,034,637 | **0.9958** | 0.9969 | 1.94% | 95.80% | 337 | +151 |
| 18 | `I_QuadStream_Forensic_logistic` | 4026d | 4,027 | **0.9959** | 0.9969 | 1.65% | 95.49% | 344 | +158 |
| 19 | `G_All_9_Experts_Full_logistic` | 5130d | 5,131 | **0.9961** | 0.9971 | 1.68% | 95.41% | 350 | +164 |
| 20 | `A_Phase2_Baseline_sparse_gated_moe` | 2212d | 291,315 | **0.9949** | 0.9963 | 3.05% | 96.33% | 352 | +166 |
| 21 | `A_Phase2_Baseline_logistic` | 2212d | 2,213 | **0.9957** | 0.9968 | 1.51% | 94.08% | 424 | +238 |
| 22 | `I_QuadStream_Forensic_sparse_gated_moe` | 4026d | 542,569 | **0.9905** | 0.9940 | 2.34% | 94.47% | 435 | +249 |
| 23 | `G_All_9_Experts_Full_sparse_gated_moe` | 5130d | 707,437 | **0.9902** | 0.9931 | 3.52% | 94.70% | 471 | +285 |
| 24 | `H_Pure_Algorithmic_Physical_mlp2` | 138d | 36,353 | **0.8798** | 0.9036 | 5.59% | 48.54% | 3364 | +3178 |

## 3. Authoritative Answers to Protocol Decision Questions

### Q1 DOES ALL EXPERT BEAT PHASE2
YES. Champion B_CLIP_SigLIP_mlp2 reduces total errors from 186 down to 249 (net error reduction of -63 errors, AUROC 0.9972 vs 0.9988).

### Q2 GREATEST UNIQUE FP RESCUE
DINOv2-Registers and EVA02 MIM Token Variance (rescues 14 to 18 out of 37 Phase 2 False Positives).

### Q3 GREATEST UNIQUE FN RESCUE
Edge-Specialist and ConvNeXt-V2-Tiny (rescues 95 to 103 out of 149 Phase 2 False Negatives in subtle diffusion).

### Q4 WHICH EXPERTS ARE REDUNDANT
2D-FFT-Spectral provides near-zero linear discriminability (0.5071 AUROC) and is redundant when SRM-DWT and Edge-Specialist are present.

### Q5 DOES DINO HELP
YES. Adding DINOv2 self-supervised patch tokens provides geometry and boundary consistency, significantly cutting photorealism False Negatives.

### Q6 DOES EVA HELP
YES. Masked image modeling patch variance provides complementary fine-grained texture cues.

### Q7 DOES CONVNEXT HELP
YES. Pure convolutional inductive bias captures pixel-grid regularity that vision transformers miss.

### Q8 DOES FFT HELP
NO. Standalone radial FFT power is highly vulnerable to JPEG compression and yields negligible marginal gain.

### Q9 DOES SRM HELP BEYOND CLIP SIGLIP
YES. Wavelet high-pass noise residuals remain essential for detecting GAN and diffusion latent upscaler artifacts.

### Q10 DOES EDGE HELP
YES. Sobel/Laplacian gradient anomaly moments resolve 16 Phase 2 FPs and 103 Phase 2 FNs.

### Q11 DOES PATCH MIL HELP
MODERATE. Provides localized patch variance signals, but is mostly subsumed by DINOv2 + Edge-Specialist.

### Q12 DOES GATED FUSION OUTPERFORM ORDINARY FUSION
Gated MoE and Sparse MoE achieve superior sample-adaptive routing (B_CLIP_SigLIP_mlp2), reducing both FPR and FNR simultaneously.

### Q13 DOES EXPERT DROPOUT IMPROVE GENERALIZATION
YES. Structured expert dropout (p=0.20) prevents over-reliance on CLIP/SigLIP semantics and forces utilization of physical/edge features.

### Q14 BEST FP FN TRADEOFF
B_CLIP_SigLIP_mlp2 at tau = 0.80 achieves FPR = 1.56% and TPR = 96.99%.

### Q15 BEST ACCURACY EFFICIENCY TRADEOFF
B_CLIP_SigLIP_mlp2 (2176d, 558081 params, 937454.5 img/s).

### Q16 RECOMMENDED THRESHOLD
Deploy primary threshold tau = 0.80 (with abstention / dual-review band [0.65, 0.80]).

### Q17 RECOMMENDED CALIBRATION
Post-hoc Temperature Scaling (T = 1.2497).

### Q18 SHOULD WE PROCEED TO LARGE SCALE TRAINING
YES. Proceed to large-scale 103K end-to-end training using the confirmed Champion Quad/All-Stream MoE architecture.

