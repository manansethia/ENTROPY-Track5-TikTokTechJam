#!/usr/bin/env python3
"""
scripts/launch_kaggle_gpu_selftest.py
Deploys a lightweight, fast GPU verification kernel to Kaggle.
Tests CUDA, VRAM, PyTorch Tensor Core MatMul, and Internet access.
"""

import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
WORKER_DIR = REPO_ROOT / "kaggle_workers" / "gpu-selftest"
WORKER_DIR.mkdir(parents=True, exist_ok=True)

PROBE_CODE = """# Kaggle Worker GPU & Network Probe
import os, sys, time, json, urllib.request
import torch

print("=" * 60)
print("=== KAGGLE WORKER HARDWARE & ENVIRONMENT PROBE ===")
print("=" * 60)

cuda_avail = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1) if cuda_avail else 0

# Torch CUDA matmul test
torch_cuda_ok = False
if cuda_avail:
    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        torch_cuda_ok = bool(y.sum().item() != 0)
    except Exception as e:
        print(f"Torch CUDA test error: {e}")

# Internet probe
internet_ok = False
try:
    with urllib.request.urlopen("https://huggingface.co", timeout=5) as r:
        internet_ok = (r.status == 200)
except Exception as e:
    internet_ok = False

print(f"CUDA_AVAILABLE = {cuda_avail}")
print(f"GPU_COUNT = {gpu_count}")
print(f"GPU_NAME = {gpu_name}")
print(f"VRAM_MB = {vram_mb}")
print(f"TORCH_CUDA = {'PASS' if torch_cuda_ok else 'FAIL'}")
print(f"INTERNET_ACCESS = {'CONNECTED' if internet_ok else 'NO_NETWORK'}")
print(f"GPU_SELF_TEST = {'PASS' if (cuda_avail and torch_cuda_ok) else 'FAIL'}")
print("=" * 60)

# Save structured probe result
result = {
    "cuda_available": cuda_avail,
    "gpu_count": gpu_count,
    "gpu_name": gpu_name,
    "vram_mb": vram_mb,
    "torch_cuda": "PASS" if torch_cuda_ok else "FAIL",
    "internet": "CONNECTED" if internet_ok else "NO_NETWORK",
    "gpu_self_test": "PASS" if (cuda_avail and torch_cuda_ok) else "FAIL"
}
with open("probe_result.json", "w") as f:
    json.dump(result, f, indent=2)
"""

with open(WORKER_DIR / "gpu-selftest.py", "w") as f:
    f.write(PROBE_CODE)

# Note: enable_gpu and enable_internet MUST be boolean true in JSON
meta = {
    "id": "doubleggunther/aigc-gpu-selftest",
    "title": "aigc-gpu-selftest",
    "code_file": "gpu-selftest.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True
}

with open(WORKER_DIR / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Pushed metadata:\n{json.dumps(meta, indent=2)}")
