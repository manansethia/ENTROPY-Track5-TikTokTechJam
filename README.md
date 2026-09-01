# AIGC Forensics — Multi-Paradigm Image Authenticity & Forensic Station

A production-grade, explainable AI-generated content (AIGC) forensic detection platform and physical investigation workstation designed for high-assurance image authenticity verification, localized AI edit localization, and cross-generator robustness.

---

## 1. System Overview

The platform is designed around three core tenets:

1. **Triple-Hybrid Champion Vision Model (~735M Parameters)**:
   - **Stream A (OpenAI CLIP ViT-L/14)**: High-level semantic composition, optical perspective, and anatomical coherence.
   - **Stream B (Google SigLIP SO400M)**: Multi-modal photorealism fidelity and fine-grained texture consistency.
   - **Stream C (Spatial-Rich Model 5x5 + Haar Wavelet Detail + ConvNeXt)**: Deterministic high-frequency spatial noise residuals and generator upsampling grid artifacts.
   - **Reliability Gating & Calibration**: Learned gating dynamically downweights frequency reliance when lossy compression or blur erases frequency traces. Calibrated via temperature scaling ($T=1.5230$) for strict enterprise operating points (FPR $\le 0.10\%$ and FPR $\le 0.01\%$).

2. **Luxury Physical Investigation Workstation**:
   - AAA-game physical investigation environment: Dark walnut wood trim, deep green felt evidence table (bottom 52% of viewport), subtle top-down roulette probability metaphor, and archival geometric background.
   - Physical evidence cards with 3D physics stacking, forensic laser scanning, realistic card deck shuffle, and physical 3-deck dealing into **REAL**, **PARTIAL-AI**, and **FULL-AIGC** piles.

3. **Pluggable Architecture & Data Honesty**:
   - `ForensicModelAdapter` interface cleanly isolates the primary detector from the frontend, allowing future model upgrades (V3, V5-CAG) without UI refactoring.
   - Genuine forensic metrics: 2D radial FFT power ratio, SRM noise residuals, Laplacian edge variance, true camera EXIF metadata, and C2PA / Content Credentials parsing.

---

## 2. Repository Layout

```text
aigc_robust_detection/
├── app/
│   ├── server.py              # Production FastAPI REST & Batch Analysis Server
│   └── static/                # Static assets & frontend bundle
├── checkpoints/
│   └── production/
│       └── final_champion_frozen_model.pt   # Triple-Hybrid Champion Checkpoint (~735M params)
├── deployment/
│   ├── cloudflare_tunnel.md   # Cloudflare Tunnel deployment guide
│   ├── portable_model.py      # Portable model definition with zero duplicate RAM overhead
│   └── schemas.py             # Pydantic request/response schemas
├── frontend/
│   ├── index.html             # Luxury Physical Forensic Workstation UI
│   ├── style.css              # Custom styling, 3D table physics, & dark walnut/felt theme
│   ├── app.js                 # 3D Deck animation, shuffle, 3-deck deal, & inspection viewer
│   └── assets/                # Roulette asset, felt texture, walnut wood, & archival SVG
├── models/
│   ├── tri_hybrid_detector.py # Tri-stream architecture definition
│   ├── srm_filters.py         # SRM 5x5 filter definitions & Haar wavelet decomposition
│   └── forensic_explainability.py # ViT Grad-CAM & Attention Rollout suite
├── server/
│   ├── forensic_adapter.py    # Swappable model adapter architecture
│   ├── provenance_engine.py   # EXIF, XMP, IPTC & C2PA extraction
│   └── spatial_engine.py      # FFT, SRM residuals, edge variance & inpainting localization
├── infer.py                   # Authoritative Hackathon CLI evaluation script
└── requirements.txt           # Python dependencies
```

---

## 3. Quickstart & Installation

### Local Web Client (Mac / Desktop)

```bash
# Clone repository
git clone <REPO_URL>
cd aigc_robust_detection

# Install lightweight dependencies
pip install -r requirements.txt

# Launch local server
python -m uvicorn app.server:app --port 8000
```
Open `http://localhost:8000` in any modern web browser.

---

## 4. Hackathon Directory Inference CLI

Run inference across any folder of images to generate the required predictions JSON:

```bash
python infer.py --input-dir ./test_images --output predictions.json
```

**Output format (`predictions.json`)**:
```json
[
  {
    "image_path": "sample_01.jpg",
    "pred": 0.0152
  },
  {
    "image_path": "sample_02.png",
    "pred": 0.9884
  }
]
```

To output detailed forensic metadata (EXIF, SHA-256, SRM energy, affected area %):
```bash
python infer.py --input-dir ./test_images --output predictions.json --detailed
```

---

## 5. Production deployment

Deploy the API only behind an authenticated private inference service. Public hosting is configured through Cloudflare Tunnel; see `deployment/cloudflare_tunnel.md`. Infrastructure hostnames, addresses, hardware, filesystem paths, and administrator access details are deliberately not stored in this public guide.

---

## 6. Dataset Governance & Locked Benchmarks

To guarantee zero data contamination, demonstration benchmarks are strictly isolated from training:

- **Held-Out Demonstration Benchmark**:
  - `COCO val2017`: 4,998 Authentic Camera Photographs.
  - `WildFake DALL-E Advanced`: 8,843 Synthetic Images.
  - *Never present in any training split or manifest.*
- **Training Corpora**: 257,755 samples balanced across 12 generator families (SD 1.5, SDXL, Midjourney, DALL-E 3, FLUX.1, GLIDE, ADM, BigGAN, StyleGAN) and 5 authentic domains (SID_Set, CIFAKE, RealPool 4K/8K, WikiArt).

---

## 7. License & Acknowledgements

- OpenAI CLIP (`ViT-L/14`) & Google SigLIP (`SO400M`) under their respective open research licenses.
- Textures & visual assets created for the AIGC Forensics project under MIT License.
