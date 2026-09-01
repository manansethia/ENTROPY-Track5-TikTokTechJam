# Final VLM Forensic & Multi-Expert Validation Report (v3)

**Generated**: 2026-08-29T06:54:44Z
**Operational Status**: `VLM_FORENSIC_OPERATIONAL = FAILED`

## 1. Operational Status Verdicts

| Gate / Component | Status | Empirical Rationale |
| :--- | :---: | :--- |
| `VLM_LOAD_VALID` | **`EXECUTED`** | Moondream2 (`2024-08-26`) loaded on `cuda:0` ($3,568.96\text{ MB}$ VRAM). |
| `VLM_FORENSIC_VALID` | **`EXECUTED`** | Direct forward passes completed on 6 real training images. |
| `VLM_STRUCTURED_OUTPUT_VALID` | **`FAILED`** | Model produces unstructured descriptive text on line-oriented schema prompts; zero keyword extraction applied. |
| `DINO_VALID` | **`EXECUTED`** | DINOv2-Registers-L forward passes executed ($1024\text{d}$ embeddings, pairwise cosine sim: $-0.0380$ to $+0.1095$). |
| `EDGE_VALID` | **`INVALID`** | Neural Edge-Specialist exhibits representation collapse ($S_{\cos} = 0.9999$ across distinct images due to uninitialized head/LayerNorm); permanently invalidated. Handcrafted Sobel gradient energy remains valid. |
| `CRITIC_VALID` | **`EXECUTED`** | 4 critic cases evaluated (`CRITIC_INDEPENDENCE = LIMITED`). |
| `COUNTERFACTUAL_VALID` | **`EXECUTED`** | Master Detector executed on original vs masked regions ($P_{\text{orig}}$, $P_{\text{masked}}$, $\Delta P$ recorded; `SPATIAL_COUNTERFACTUAL = UNAVAILABLE`). |
| `VLM_FORENSIC_OPERATIONAL` | **`FAILED`** | **BLOCKED**: Gated on `VLM_STRUCTURED_OUTPUT_VALID = FAILED` and `EDGE_VALID = INVALID`. |

## 2. Invalidation of Neural Edge Specialist

- **Finding**: `EdgeArtifactFeatureExtractor (256d)` has representation collapse ($S_{\cos} = 0.9999$, $\|x\|_2 = 15.9697$).
- **Action**: Permanently marked **`INVALID`**. Excluded from feature fusion and feedback learning.
- **Approved Physical Signal**: `HANDCRAFTED_FORENSIC_FEATURES` (Sobel gradient magnitude mean, Laplacian residuals, 2D-FFT ratio, SRM).

## 3. DINOv2 Representation Diversity

- **Checkpoint SHA-256**: `edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b`
- **Pairwise Cosine Similarity**: Ranging from $-0.0380$ to $+0.1095$ across distinct images, confirming orthogonal representation capability.
- **Pairwise Euclidean Distances**: $30.67$ to $34.09$.

## 4. Counterfactual Master Detector Evidence

- **REAL_SAMPLE_1_WIKIART**: $P(\text{AIGC})_{\text{orig}} = 0.724739 \to P(\text{AIGC})_{\text{masked}} = 0.719998$, $\Delta P = -0.004741$ (Spatial Counterfactual: `UNAVAILABLE`).
- **AIGC_SAMPLE_1_QUALITY_PARADOX**: $P(\text{AIGC})_{\text{orig}} = 0.718321 \to P(\text{AIGC})_{\text{masked}} = 0.716894$, $\Delta P = -0.001427$ (Spatial Counterfactual: `UNAVAILABLE`).
