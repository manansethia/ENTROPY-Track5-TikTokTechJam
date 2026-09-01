# Deployment & Hardware Integration Documentation
# AIGC Robust Forensic Vision Detector (<2B Parameters)

---

## 1. System Architecture Overview

The deployment suite provides a model-agnostic, multi-backend inference service with sub-millisecond preprocessing and calibrated multi-tier confidence routing.

```
                    [Inbound Client Image / API Request]
                                     │
                                     ▼
                     [FastAPI High-Throughput Server]
                             (deployment/api.py)
                                     │
                                     ▼
                  [Deterministic Preprocessing Pipeline]
                          (deployment/preprocess.py)
                   • Bicubic Resampling to 224x224
                   • Normalized Tensor [0.4814, 0.4578, 0.4082]
                                     │
                                     ▼
                     [Forensic Inference Engine]
                          (deployment/inference.py)
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
    [Macro-Semantic Stream]                         [Micro-Forensic Stream]
    • CLIP ViT-L/14 (768-d)                         • GPU Wavelet SRM Residuals
    • SigLIP SO400M (1152-d)                        • Haar DWT Detail Sub-bands
             └───────────────────────┬───────────────────────┘
                                     ▼
                       [Fused Embedding Normalization]
                                     │
                                     ▼
                   [Calibrated Temperature Scaling (T=1.247)]
                                     │
                                     ▼
                   [Multi-Tier Low-FPR Decision Routing]
            • Standard Balanced Mode:      Threshold = 0.5000 (FPR ~1.00%)
            • Ultra-Reliable Enterprise:   Threshold = 0.9984 (FPR <= 0.10%)
            • Zero-False-Alarm Mode:       Threshold = 0.9998 (FPR <= 0.01%)
                                     │
                                     ▼
             [Structured JSON Prediction + Forensic Attribution]
```

---

## 2. API Endpoints Reference

### 2.1 `POST /v1/predict`
Single image classification with optional micro-forensic Fourier and Laplacian attribution.

**Parameters:**
- `file` (Multipart Form Upload) or `image_base64` (JSON Body)
- `threshold_mode`: `standard` | `low_fpr_10` | `low_fpr_05` | `low_fpr_01` | `low_fpr_001`
- `include_forensic_breakdown`: `true` | `false`

**Sample Response:**
```json
{
  "success": true,
  "probability_aigc": 0.002145,
  "raw_logit": -5.124312,
  "predicted_class": "AUTHENTIC_REAL",
  "is_aigc": false,
  "confidence_tier": "HIGH_CONFIDENCE_REAL",
  "threshold_used": 0.998450,
  "threshold_mode": "low_fpr_01",
  "latency_ms": 14.82,
  "device_used": "cuda:0",
  "model_version": "ScientificVisionDetector-ConfigA",
  "model_sha256": "0cde8de29d2b2be3a4ec8feab78ef9292871806bab035dd127051de6a4d2633e",
  "forensic_breakdown": {
    "fft_high_frequency_ratio": 0.0824,
    "srm_residual_energy": 2.145,
    "laplacian_variance": 218.4,
    "inconsistency_status": "CLEAN"
  }
}
```

### 2.2 `POST /v1/predict/batch`
High-throughput batched inference (up to 128 images per request).

### 2.3 `GET /v1/metadata`
Returns frozen model parameters, parameter hash, input resolution, and pre-registered threshold tables.

### 2.4 `GET /health`
System health telemetry including active device, VRAM allocated, and host RAM.

---

## 3. Hardware Support Matrix

| Hardware Target | Execution Backend | Precision | Cold-Start | Batch=1 Latency | Throughput (Batch=32) | Memory Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 3050 (6GB)** | PyTorch CUDA Autocast | BFloat16 / FP16 | $\sim 450\text{ ms}$ | $\mathbf{12.4\text{ ms}}$ | $\mathbf{148.5\text{ img/s}}$ | $1.85\text{ GB}$ VRAM |
| **x86_64 Multi-Core CPU** | PyTorch CPU (MKL/OpenMP) | Float32 | $\sim 380\text{ ms}$ | $\mathbf{48.2\text{ ms}}$ | $\mathbf{32.4\text{ img/s}}$ | $850\text{ MB}$ RAM |
| **Apple Silicon (M-Series)** | PyTorch MPS / CoreML | Float16 | $\sim 420\text{ ms}$ | $\mathbf{18.6\text{ ms}}$ | $\mathbf{85.0\text{ img/s}}$ | $1.20\text{ GB}$ Unified |
| **ONNX Runtime (Generic)** | CPU / TensorRT Provider | FP16 / FP32 | $\sim 300\text{ ms}$ | $\mathbf{14.1\text{ ms}}$ | $\mathbf{120.0\text{ img/s}}$ | $750\text{ MB}$ RAM |

---

## 4. Running the Service on Buildabot

```bash
# 1. Activate Environment
source /home/manan/.venvs/aigc-detector/bin/activate

# 2. Run Standalone Benchmark
python -m deployment.benchmark --device cpu --iterations 30
python -m deployment.benchmark --device cuda:0 --iterations 50

# 3. Launch High-Throughput REST API & Web UI (Local/Private Worker Mode)
uvicorn deployment.api:app --host 127.0.0.1 --port 8000 --workers 1
```
