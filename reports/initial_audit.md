# Initial Audit Report: AIGC Robust Detection Project

**Date:** 2026-08-28  
**Role:** Lead ML Research Engineer  
**Status:** Audit Complete — Ready for Remote Execution & Experimentation  

---

## 1. Current Architecture Overview

The codebase implements a **Tri-Stream Spatial-Frequency Hybrid AIGC Detector** (`MasterEnsembleDetector` in `models/tri_hybrid_detector.py`):
1. **Stream 1 (Macro-Semantic):** Frozen `openai/clip-vit-large-patch14` (~304M params) extracting semantic/compositional embeddings (768-dim).
2. **Stream 2 (Regional Semantic):** Frozen `google/siglip-base-patch16-224` (~86M params) extracting regional visual features (768-dim).
3. **Stream 3 (Micro-Forensic Frequency):** 
   - Fixed 3x3 SRM High-Pass Filter (`models/srm_filters.py`).
   - 2D Haar Discrete Wavelet Transform (DWT) extracting 3 detail sub-bands (LH, HL, HH) across 3 RGB channels (9 channels).
   - Trainable `timm/convnext_tiny` (~28M params, modified `in_chans=9`).
4. **Dynamic Reliability Gate:** A 2-layer MLP (`Linear(768, 128) -> GELU -> Linear(128, 3) -> Softmax`) computing adaptive weights across the 3 streams.
5. **Classification Head:** Linear projection + GELU + Dropout(0.3) + Linear classifier outputting scalar AIGC logits.

**Parameter Summary:**
- Total Parameters: **~418.5M**
- Trainable Parameters: **~29.2M** (ConvNeXt-Tiny + Projections + Gating + Classifier)
- Frozen Parameters: **~389.3M** (CLIP ViT-L/14 + SigLIP-Base)
- Rule Check: **Strictly < 2,000,000,000 parameters (PASS — utilizing ~21% of budget)**.

---

## 2. Current Repository Structure

```text
aigc_robust_detection/
├── .env                              # Remote host & API credentials
├── .gitignore                        # Git exclusion rules
├── MANIFEST_FINAL.txt                # Repository manifest
├── README.md                         # Project overview and instructions
├── requirements.txt                  # Python dependencies
├── configs/
│   └── train_config.yaml             # Hyperparameters, augmentation rates, paths
├── models/
│   ├── __init__.py                   # Model exports
│   ├── srm_filters.py                # SRM convolution & 2D Haar DWT
│   └── tri_hybrid_detector.py        # MasterEnsembleDetector tri-stream architecture
├── scripts/
│   ├── data.py                       # Dataset loading & indexing
│   ├── augmentations.py              # Perturbation implementations & training corruptions
│   ├── transforms.py                 # Normalization & tensor preprocessing
│   ├── train_detector.py             # PyTorch AMP training loop
│   ├── run_inference.py              # Batch directory inference -> results.json
│   ├── evaluate_robustness.py        # 15-condition robustness benchmark evaluator
│   ├── verify_model.py               # Instantiation and parameter ceiling verification
│   └── setup_server_env.sh           # Legacy Ubuntu/Conda setup script (to be replaced)
├── server/
│   ├── README_SERVER_SETUP.md        # Server environment instructions
│   ├── memory/
│   │   ├── memory_guard.py           # CUDA memory tracking, snapshotting, hard cleanup
│   │   └── expert_worker.py          # Subprocess isolation template
│   └── scripts/
│       ├── 00_hardware_audit.sh      # Hardware & GPU environment audit
│       ├── 01_prepare_server.sh      # Directory tree & venv initialization
│       ├── 02_install_ml_stack.sh    # Fedora driver-matched PyTorch + ML stack
│       ├── 03_download_model_pool.sh # Model downloader (CLIP, SigLIP2, DINOv2, AIDE, DDA)
│       ├── 04_download_datasets_full.sh # Dataset downloader (CF-Small, SID, WildFake)
│       ├── 05_lock_validation.sh     # Validation benchmark isolation locker
│       ├── 05_stream_community_forensics.py # Streaming CF-Small materializer
│       ├── 06_verify_assets.sh       # Asset integrity check
│       ├── 07_benchmark_model_pool.sh # Model pool inventory & benchmarking
│       ├── 08_gpu_smoke_test.sh      # CUDA matmul smoke test
│       ├── 10_storage_guard.sh       # Storage threshold check
│       ├── 11_memory_diagnostics.sh  # VRAM diagnostics
│       ├── 12_install_optional_stack.sh # Optional offload stack
│       ├── 13_system_dependencies_fedora.sh # DNF system packages
│       └── 14_preflight_all.sh       # Preflight validation runner
├── tools/
│   └── count_parameters.py           # Standalone Hugging Face parameter counter
├── docs/
│   ├── AGENT_HANDOFF_MASTER.md       # Master agent handoff
│   ├── ARCHITECTURE.md               # Architecture details
│   ├── DATASET_GUIDE.md              # Dataset descriptions & isolation rules
│   ├── DEMO_SCRIPT.md                # Video demo script
│   ├── DEPLOYMENT_AND_DOWNLOAD_PLAN.md # Server download plan
│   ├── DEVPOST_DESCRIPTION.md        # Devpost submission draft
│   ├── ERROR_ANALYSIS.md             # Error analysis notes
│   ├── EVALUATION.md                 # Evaluation protocol
│   ├── FINAL_BUILD_CHECKLIST.md      # Final verification checklist
│   ├── IMPLEMENTATION_NOTES.md       # Technical implementation notes
│   ├── INSTALL_AND_RUN.md            # Installation instructions
│   ├── MASTER_RESEARCH_AND_KNOWLEDGE_BASE.md # Comprehensive knowledge base
│   ├── MAX_PERFORMANCE_PLAN.md       # Maximum performance roadmap
│   ├── MODEL_SELECTION_MATRIX.md     # Model comparison scorecard
│   ├── RESEARCH_SOURCES_AND_RATIONALE.md # Research background & citations
│   ├── SUBMISSION_CHECKLIST.md       # Submission verification items
│   └── TOOLS_AND_FRAMEWORK_DECISION.md # Framework decisions
└── reference/
    ├── AIGC_Image_Detection_Pipeline.pdf # Source pipeline design document
    └── hackathon_brief_and_original_spec.md # Original hackathon brief
```

---

## 3. Known Problems & Critical Deficiencies Identified

1. **DataLoader Collation Error Risk in `scripts/data.py`:**
   - In `AIGCDataset.__getitem__`, images of varying native resolutions are converted to `torch.Tensor` without spatial resizing.
   - `DataLoader(batch_size > 1)` will throw a `RuntimeError` during batch collation (`stack expects each tensor to be equal size`).
   - *Fix:* Ensure all images are resized or cropped to standard resolution (e.g. 224x224) prior to tensor conversion in `data.py`.

2. **Model Name Consistency (`SigLIP` vs `SigLIP2`):**
   - `models/tri_hybrid_detector.py` defaults to `google/siglip-base-patch16-224` (SigLIP v1), whereas modern research recommends `google/siglip2-base-patch16-224` or `google/siglip2-large-patch16-384`.
   - *Fix:* Update default configs and model constructor to support both SigLIP v1 and SigLIP2 checkpoints dynamically.

3. **Legacy Ubuntu Script in `scripts/setup_server_env.sh`:**
   - Uses `apt-get update && apt-get install` and assumes Ubuntu/Conda. The target server is Fedora Linux.
   - *Fix:* Direct server setup through `server/scripts/` (DNF-based, venv-based) and update `scripts/setup_server_env.sh` to be OS-agnostic.

4. **Lack of Validation Path Guard in Training Script:**
   - Although `docs/DATASET_GUIDE.md` specifies strict isolation, `scripts/train_detector.py` did not programmatically assert that input directories exclude `validation_LOCKED`.
   - *Fix:* Add an explicit programmatic assertion in `data.py` raising an exception if `validation_LOCKED` appears in any dataset path.

5. **SSH Connection Authentication:**
   - Remote host `buildabot.lykoi-typhon.ts.net` (`100.69.97.120`) is reachable via Tailscale network, but password/publickey authentication is required to execute remote shell commands.
   - *Fix:* Configure SSH key or password access.

---

## 4. Hardware Findings & Storage Hierarchy

- **Target Machine:** `buildabot.lykoi-typhon.ts.net` (`100.69.97.120`)
- **OS:** Fedora Linux
- **CPU:** Intel Core i5 12th Gen (12 vCPUs)
- **RAM:** 32 GB RAM
- **GPU:** NVIDIA GeForce RTX 3050 (6 GB VRAM, Ampere GA106/GA107, Compute Capability 8.6)
- **Primary NVMe (`/`):** ~475 GB available for OS, Python virtualenv (`~/.venvs/aigc-detector`), active git repository, hot cache.
- **Secondary HDD (`/mnt/ai-storage`):** ~931.5 GB available at `/mnt/ai-storage/aigc_data/` for:
  - `models/`: Pretrained foundation encoders & baseline checkpoints
  - `datasets/`: Community Forensics, SID_Set, WildFake training, GenImage
  - `validation_LOCKED/`: Isolated COCO val2017 & WildFake DALL-E Advanced
  - `features/`: Cached feature embeddings
  - `checkpoints/`: Trained model weights & training states
  - `hf_cache/`: Hugging Face and ModelScope cache

---

## 5. Recommended Fixes & Next Actions

1. **Refactor `scripts/data.py` and `scripts/transforms.py`:**
   - Implement deterministic resizing to fixed canvas before batch collation.
   - Add programmatic `validation_LOCKED` path assertion guard.
2. **Setup and Configure Remote Server Environment:**
   - Execute `server/scripts/00_hardware_audit.sh` and `01_prepare_server.sh`.
   - Install driver-matched PyTorch (`cu124` / `cu126` / `cu128`) and verify CUDA via `08_gpu_smoke_test.sh`.
3. **Model Acquisition & Parameter Validation:**
   - Download model pool (`CLIP ViT-L/14`, `SigLIP2-Large/Base`, `DINOv2-Large`, `ConvNeXt-Tiny`, `AIDE`, `DDA`).
   - Programmatically verify parameter counts for every candidate to guarantee `< 2,000,000,000` parameters.
4. **Dataset Acquisition & Feature Caching:**
   - Materialize streaming Community Forensics-Small slice and SID_Set.
   - Isolate locked benchmark files under `/mnt/ai-storage/aigc_data/validation_LOCKED/`.
   - Pre-compute and cache frozen foundation embeddings to maximize RTX 3050 6GB throughput.
5. **Systematic Experimentation & Benchmarking:**
   - Train baseline fusion heads and evaluate against all 15 challenge perturbations.
   - Run error analysis on representative false positives/negatives.
   - Generate submission artifacts: `inference.py`, `check_parameter_limit.py`, `reports/final_robustness_report.md`, `reports/final_metrics.json`.
