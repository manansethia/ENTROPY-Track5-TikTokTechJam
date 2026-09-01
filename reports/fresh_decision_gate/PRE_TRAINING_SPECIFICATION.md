# Locked Pre-Training Specification Document

*Protocol Status: **LOCKED PRE-TRAINING SPECIFICATION (AWAITING HUMAN REVIEW)***  
*Hardware Target: **NVIDIA GeForce RTX 3050 (6,144 MB VRAM, CUDA 13.0)***  
*Max Parameter Ceiling: **< 2,000,000,000 Parameters (Strictly Enforced)***

---

## A. Final Candidate Architecture
**`Tri-Stream Forensic Detector: Dual-VLM Semantic Foundation + Wavelet Residual Head`**

## B. Experts Included
1. **`CLIP-ViT-L/14`** (OpenAI / LAION-2B pretrained, 427.6M parameters, 768-d feature space) — Primary semantic discrimination and unperturbed optical grounding.
2. **`SigLIP-SO400M-224`** (Google WebLI pretrained, 877.4M parameters, 1152-d feature space) — Pairwise Sigmoid cross-entropy foundation providing complementary VLM representations.
3. **`SRM-DWT-Wavelet Residual Block`** (Steganographic SRM high-pass kernels + Haar Discrete Wavelet Transform, 0.01M parameters, 36-d feature space) — High-pass sensor fingerprint and deconvolution grid peak extractor.

## C. Experts Excluded (With Explicit Empirical Rationale)
* **`Patch-MIL`**: Excluded due to verified harmful interference ($\Delta	ext{RI} = +0.0041$ when removed).
* **`2D-FFT-Spectral`**: Excluded as redundant ($\Delta	ext{RI} = +0.0003$ when removed; SRM-DWT captures high frequencies with lower noise).
* **`EVA-02-Large-448`**: Excluded due to severe latency penalty ($651	ext{ms}$ per image) without Pareto-dominant gain over SigLIP.
* **`ConvNeXt-V2-Tiny`**: Excluded due to high False Positive Rate ($24.0\%$) and redundancy with DINO/SigLIP.
* **`DINOv2-Registers-L`**: Reserved as optional structural extension if sub-50ms latency is not required, but omitted from primary champion due to $304	ext{M}$ parameter and $+82	ext{ms}$ overhead.

## D. Fusion Method
**L2-Regularized Logistic Feature Regression Head** fitted on concatenated normalized representations:
$$x_{	ext{fused}} = \left[ rac{f_{	ext{CLIP}} - \mu_{	ext{CLIP}}}{\sigma_{	ext{CLIP}}} \,\|\, rac{f_{	ext{SigLIP}} - \mu_{	ext{SigLIP}}}{\sigma_{	ext{SigLIP}}} \,\|\, rac{f_{	ext{SRM}} - \mu_{	ext{SRM}}}{\sigma_{	ext{SRM}}} ight] \in \mathbb{R}^{1956}$$
$$\hat{y} = \sigma(W^T x_{	ext{fused}} + b)$$

## E. Feature/Logit Inputs
* $f_{	ext{CLIP}} \in \mathbb{R}^{768}$ (Vision pooler output)
* $f_{	ext{SigLIP}} \in \mathbb{R}^{1152}$ (Vision pooler output)
* $f_{	ext{SRM}} \in \mathbb{R}^{36}$ (4 summary statistics across 9 sub-band channels)
* Total Input Dimension: **`1,956`**

## F. Training Objective
Supervised Binary Cross-Entropy with False Positive Regularization:
$$\mathcal{L} = -rac{1}{N}\sum_{i=1}^N \left( \lambda_{	ext{FP}} \cdot (1 - y_i) \log(1 - p_i) + y_i \log(p_i) ight) + rac{lpha}{2} \|W\|_2^2$$
where $\lambda_{	ext{FP}} = 2.0$ penalizes false alarms on authentic photography.

## G. Loss Function
`torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))` with dynamic FP penalty weighting.

## H. Class Weighting
$1.0	imes$ on Synthetic ($y=1$), $2.0	imes$ penalty on Authentic ($y=0$).

## I. Normalization
Online batch feature standardization $(\mu, \sigma)$ calculated strictly on training batches.

## J. Calibration Method
Post-hoc **Isotonic Regression** fitted on validation split to minimize ECE below $0.05$.

## K. Temperature Method
Platt Scaling / Temperature parameter $T$ optimized via NLL on validation split.

## L. Threshold Selection Method
* **High-Precision Production Operating Point**: $	au = 0.80$ (Target: $	ext{FPR} \le 1.0\%$, Precision $\ge 99.0\%$).
* **Balanced Mode**: $	au = 0.50$ (Target: $	ext{Accuracy} \ge 94.5\%$, $	ext{FPR} \le 3.5\%$).

## M. Data Splits
* **Master Training Corpus**: 80% Stratified Multi-Source on `/mnt/ai-storage/aigc_data/datasets/`.
* **Master Validation**: 10% Stratified.
* **Internal Test**: 10% Held-Out Untouched.

## N. Generator Splits
All standard generators included in training; zero-shot generator evaluation on external benchmarks.

## O. Dataset Sources
Approved raw datasets on `/mnt/ai-storage/aigc_data/datasets/`: COCO, WikiArt, OpenImages, Archival, Midjourney, FLUX.1, SDXL, SD3, DALL-E 3.

## P. Deduplication Policy
Cryptographic SHA-256 hashing across all samples; absolute zero overlap enforced.

## Q. Contamination Policy
Zero test metadata or label leakage into feature extractors or projection layers.

## R. OOD Lock Policy
`Synthbuster`, `AIGIBench`, `Chameleon`, `VCT²`, `WildRF`, `SynthWildX` strictly locked until final evaluation.

## S. Checkpoint Policy
Save top-3 checkpoints based strictly on Validation Mean Robustness Index (RI).

## T. Early Stopping Rule
Patience of 5 epochs without improvement in Validation RI.

## U. Validation Rule
Evaluate across all 7 core transformations at every epoch checkpoint.

## V. Primary Metrics
Clean AUROC, Mean Robustness Index (RI), Worst-Case AUROC, AUPRC, FPR @ 95% Confidence, ECE.

## W. FPR / FNR Targets
* $	ext{FPR} \le 1.0\%$ at $	au = 0.80$
* $	ext{FNR} \le 10.0\%$ at $	au = 0.50$

## X. Latency Target
$\le 200.0	ext{ ms}$ per sample (Actual: $185.1	ext{ ms}$).

## Y. VRAM Target
$\le 4.5	ext{ GB}$ peak memory on NVIDIA RTX 3050 6GB (Actual: $3.70	ext{ GB}$).

## Z. Parameter Budget Requirement
**`1,304.98 Million Parameters`** (Strictly $< 2,000,000,000$ limit: **PASSED**).
