# Master Knowledge Base: Robust Image-Level AIGC Detection & OOD Generalization

---

## 1. Architecture & Heterogeneous Multi-Domain Fusion

### 1.1 Model Topology (Config A — 31.94M Trainable Parameters)
The production detector fuses heterogeneous visual foundation representations with deterministic spatial-frequency forensic features:
1. **Macro-Semantic Invariant Visual Encoders**:
   - **OpenAI CLIP ViT-L/14** (224x224, 768-d projection): High-level semantic composition, lighting geometry, anatomical coherence.
   - **Google SigLIP SO400M** (224x224, 1152-d projection): Fine-grained multi-modal representation, texture consistency, photorealism fidelity.
2. **Micro-Forensic Spatial-Frequency Stream**:
   - **GPU-Accelerated Spatial Rich Model (SRM)**: 2D linear convolution filters isolating high-frequency noise residuals $R(x, y) = I(x, y) * K_{\text{SRM}}$.
   - **Haar Discrete Wavelet Detail Sub-bands**: Multi-scale detail isolation across horizontal ($LH$), vertical ($HL$), and diagonal ($HH$) high-frequency components.
3. **Multi-Aspect Evidence Head**:
   - 5-dimensional multi-label evidence projection head over fused embeddings:
     - `high_frequency_spectral_anomaly` (2D FFT radial power distribution)
     - `srm_noise_residual_inconsistency` (SRM deterministic residual energy)
     - `laplacian_edge_boundary_anomaly` (Edge variance / boundary discontinuity)
     - `texture_oversmoothing_inconsistency` (Diffusion plastic skin / brushstrokes)
     - `compression_resampling_artifact` (Grid / sinc interpolation residues)

---

## 2. Governed Dataset & OOD Coverage Engineering

### 2.1 Storage Inventory & Targeted Expansion Audit
A comprehensive audit of local storage (`/mnt/ai-storage/aigc_data` and `/home/manan/aigc_robust_detection`) established:
- **Strongly Represented Families**: Stable Diffusion 1.4/1.5, SDXL, Midjourney v5/v6, DALL-E 3, StyleGAN2/3.
- **Identified Critical Gaps**: GLIDE, ADM, BigGAN, VQDM, Wukong.
- **Governed Public Ingestion**: Downloaded and verified 14 shards of `TheKernel01/Tiny-GenImage` (CC BY-NC-SA 4.0). Extracted **13,500 deduplicated samples** ($2,000$ per missing generator family $+ 3,500$ ImageNet authentic real photos).
- **Locked Benchmark Isolation**: Verified $0$ SHA-256 hash collisions with locked benchmarks (Synthbuster, AIGIBench, Chameleon, VCT2, WildRF, SynthWildX, Internal Test).

### 2.2 Governed Manifest Split Allocation (`ood_remediation_manifest_v1.jsonl`)
- **TRAIN Pool**: $257,755$ images (balanced across 12 generator families, 5 real domains, varied resolutions).
- **DEV Split**: $10,000$ images (strictly frozen, source-disjoint validation).
- **CAL Split**: $4,000$ images (temperature calibration).
- **FORENSIC_EXPLANATION_VALIDATION_POOL**: $50$ isolated approved samples reserved exclusively for qualitative explanation validity audits.

---

## 3. Pseudo-OOD Generator & Real Domain Holdout Suite

To measure generalization without contaminating locked external benchmarks, 7 source-disjoint holdouts were established:

| Holdout Fold | Domain / Generator | Baseline AUROC | Baseline TPR @ 0.10% FPR |
| :--- | :--- | :---: | :---: |
| **Fold 1: SDXL_Midjourney** | Advanced Diffusion | $0.999566$ | $99.41\%$ |
| **Fold 2: SID_LatentDiffusion** | Subtle Photorealistic Latent Diffusion | $0.997138$ | $\mathbf{90.05\%}$ *(Identified Worst-Case)* |
| **Fold 3: Quality_Paradox** | Ultra-High-Fidelity Photorealism | $0.999467$ | $98.63\%$ |
| **Fold 4: Diverse_Synthetics** | Legacy Diffusion & GANs | $0.999291$ | $98.28\%$ |
| **Real Fold 1: WikiArt** | Complex Fine Art & Brushstrokes | $0.999565$ | $98.30\%$ |
| **Real Fold 2: COCO** | Natural Photography & Dense Scenes | $0.999103$ | $97.86\%$ |
| **Real Fold 3: Natural SID** | Pristine Sensor Captures | $0.997654$ | $95.68\%$ |
| **Macro Summary** | **All Generator Holdouts** | $\mathbf{0.998865}$ | $\mathbf{90.05\%}$ *(Worst-Family TPR)* |

---

## 4. Remediation Candidate Benchmark Suite: REM-A vs REM-B vs REM-C

### 4.1 Candidate Component Distinction
To isolate experimental variables, the remediation candidates systematically decompose training factors:
1. **Source / Generator Balancing (REM-A, REM-B, REM-C)**: Uniform generator-family batch stratification ensuring rare generators (GLIDE, ADM, BigGAN, VQDM, Wukong) receive equal gradient updates.
2. **Invariant Data Augmentation (REM-B, REM-C)**: Dynamic aspect-ratio cropping ($p=0.5$), JPEG compression sweeps $Q \in [40, 95]$ ($p=0.5$), Gaussian blur/sharpen $\sigma \in [0.5, 2.0]$ ($p=0.3$), and color jitter ($p=0.3$).
3. **Hard-Case Mining Curriculum (REM-C)**: Mined hard False Positives (real samples with $P(\text{AI}) > 0.40$) and hard False Negatives (AIGC samples with $P(\text{AI}) < 0.60$) up-weighted ($2.0\times$) in the training curriculum.

### 4.2 Comparative Empirical Matrix

| Model / Candidate | Strategy & Strategy Components | DEV Acc | DEV FP / FN | DEV AUROC | DEV TPR @ 0.10% FPR | DEV TPR @ 0.01% FPR | Edge Acc | Worst-Gen (SID) TPR @ 0.1% | Decision Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PRODUCTION_BASELINE** | Frozen Champion (Baseline) | $99.12\%$ | $33 / 55$ | $0.998975$ | $97.42\%$ | $96.28\%$ | $98.54\%$ | $90.38\%$ | Preserved |
| **REM-A** | Balanced Sampling Only (Ep 3) | $\mathbf{99.21\%}$ | $\mathbf{29 / 50}$ | $\mathbf{0.999180}$ | $\mathbf{98.52\%}$ | $\mathbf{97.54\%}$ | $\mathbf{98.78\%}$ | $\mathbf{94.94\%}$ | **`CHAMPION_REM_A`** |
| **REM-B** | Balanced + Invariant Aug (Ep 1) | $98.86\%$ | $30 / 84$ | $0.999213$ | $94.36\%$ | $86.44\%$ | $98.24\%$ | $80.75\%$ | Regressed |
| **REM-B** | Balanced + Invariant Aug (Ep 2) | $98.94\%$ | $43 / 63$ | $0.999227$ | $94.62\%$ | $91.40\%$ | $98.31\%$ | $78.63\%$ | Regressed |
| **REM-B** | Balanced + Invariant Aug (Ep 3) | $98.98\%$ | $38 / 64$ | $0.999230$ | $95.10\%$ | $92.20\%$ | $98.38\%$ | $81.20\%$ | **`REM_B_REJECTED`** |
| **REM-C** | Balanced + Aug + Hard Curriculum | *Pending* | — | — | — | — | — | — | Pre-Registered |

### 4.3 Scientific Analysis: Invariance vs Forensic Sensitivity
- **Observed Empirical Fact**: Despite training loss falling monotonically ($0.1116 \to 0.0692 \to 0.0620$), REM-B suffered a noticeable drop in low-FPR operation (TPR @ $0.10\%$ fell from $97.42\% \to 94.62\%$) and worst-case holdout detection (Fold SID LDM fell from $90.38\% \to 78.63\%$).
- **Working Hypothesis**: Heavy JPEG compression ($Q \in [40, 70]$) and spatial blurring artificially attenuate fine-grained spatial and frequency noise residuals that the model relies upon to identify subtle photorealistic diffusion artifacts. This hypothesis is tracked as an active research question pending controlled causal ablation.
- **Selection Rule**: Training loss is rejected as a decision metric. If REM-C does not outperform `CHAMPION_REM_A` across the combined empirical validation suite, `CHAMPION_REM_A` is permanently retained as the remediation champion.

---

## 5. Information-Theoretic Forensic Feedback Validation Gate

### 5.1 Differentiable Loss Formulation
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Class}}(z_{\text{class}}, y_{\text{true}}) + \alpha \cdot \mathcal{L}_{\text{Evidence}}(z_{\text{evidence}}, e_{\text{target}}) + \beta \cdot \mathcal{L}_{\text{Counterfactual}}(\Delta P, \Delta P_{\text{target}})$$

Where:
- **$\mathcal{L}_{\text{Class}}$**: Binary cross-entropy preserving **100% authoritative ground-truth labels** ($y_{\text{true}} \in \{0, 1\}$).
- **$\mathcal{L}_{\text{Evidence}}$**: 5-dimensional multi-label BCE over structured forensic tags supervised by Moondream2 + deterministic signal verification ($\alpha = 0.50$).
- **$\mathcal{L}_{\text{Counterfactual}}$**: $\text{SmoothL1}((P_{\text{orig}} - P_{\text{pert}}), \Delta P_{\text{target}})$ aligning model sensitivity with verified localized causal anomalies ($\Delta P_{\text{target}} = \pm 0.35$, $\beta = 0.25$).

### 5.2 3-Condition Information-Theoretic Gradient Ablation
- **Condition A (Pure Classification)**: $\mathcal{L}_{\text{A}} = \text{BCE}(z, y)$. Baseline gradient $\nabla \theta_A$.
- **Condition B (Scalar Confidence Weighting)**: $\mathcal{L}_{\text{B}} = (1 + \alpha w) \text{BCE}(z, y)$. Proves $\cos(\nabla \theta_A, \nabla \theta_B) \equiv 1.0000$ (pure collinear scalar scaling with zero new information).
- **Condition C (Full Evidence + Causal CF)**: $\mathcal{L}_{\text{C}} = \mathcal{L}_{\text{Class}} + \alpha \mathcal{L}_{\text{Evidence}} + \beta \mathcal{L}_{\text{CF}}$. Proves $\cos(\nabla \theta_A, \nabla \theta_C) < 1.0000$ (injects genuinely non-collinear, orthogonal evidence gradients into the optimization trajectory).

### 5.3 Hardware Safety Protocol (RTX 3050 6GB)
Strict sequential GPU memory offloading:
1. Detector inference & mining on GPU $\to$ Offload to CPU (`empty_cache()` $\to$ $0.1\text{ GB}$ VRAM).
2. Moondream2 FP16 on GPU $\to$ Generate structural hypotheses $\to$ Delete & offload (`empty_cache()` $\to$ $0.1\text{ GB}$ VRAM).
3. Deterministic verification on CPU NumPy/SciPy ($0\text{ GB}$ VRAM).
4. Detector reload to GPU $\to$ Compute $\mathcal{L}_{\text{Total}} \to \text{backward}() \to \text{step}()$ ($2.8\text{ GB}$ VRAM).
*Peak VRAM strictly bounded at $\leq 3.6\text{ GB} / 6.14\text{ GB}$.*
