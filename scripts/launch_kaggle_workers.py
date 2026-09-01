#!/usr/bin/env python3
"""
scripts/launch_kaggle_workers.py
Launches independent Kaggle GPU workers using the verified API token.
Sets up dedicated kernels for Worker 1 to Worker 5.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
ENV_PATH = REPO_ROOT / ".env"

if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

# Ensure token is set
os.environ["KAGGLE_API_TOKEN"] = "KGAT_5be018cf2a3dae2318f6980b6ce0631e"

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

WORKERS_DIR = REPO_ROOT / "kaggle_workers"
WORKERS_DIR.mkdir(parents=True, exist_ok=True)

WORKERS_SPEC = [
    {
        "id": "worker-1-spai-benchmark",
        "title": "AIGC Robustness Worker 1 - SPAI Benchmark",
        "code": """# Worker 1: SPAI / TFG-Model Benchmark
import os, sys, time, json
import torch
print("=== KAGGLE WORKER 1: SPAI / TFG BENCHMARK ===")
print("CUDA Available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
os.system("pip install -q timm yacs albumentations ftfy filetype lmdb")
os.system("git clone https://huggingface.co/aminasifar1/TFG-model /tmp/spai_tfg")
sys.path.insert(0, "/tmp/spai_tfg")
os.chdir("/tmp/spai_tfg")
from inference import EndpointHandler
from PIL import Image
handler = EndpointHandler("/tmp/spai_tfg")
print("SPAI initialized successfully on Kaggle GPU!")
"""
    },
    {
        "id": "worker-2-cf-benchmark",
        "title": "AIGC Robustness Worker 2 - CommunityForensics Benchmark",
        "code": """# Worker 2: CommunityForensics ViT-Small Benchmark
import os, sys, time, json
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
print("=== KAGGLE WORKER 2: COMMUNITY FORENSICS 21.8M BENCHMARK ===")
print("CUDA Available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
model_id = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id).cuda()
model.eval()
print("CommunityForensics ViT-Small (21.8M) initialized successfully on Kaggle GPU!")
"""
    }
]

def push_worker_kernel(spec):
    worker_id = spec["id"]
    title = spec["title"]
    code_content = spec["code"]
    
    kernel_dir = WORKERS_DIR / worker_id
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = kernel_dir / f"{worker_id}.py"
    with open(script_path, "w") as f:
        f.write(code_content)
        
    meta = {
        "id": f"doubleggunther/{worker_id}",
        "title": title,
        "code_file": f"{worker_id}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true"
    }
    
    meta_path = kernel_dir / "kernel-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Pushing Kaggle Kernel: {worker_id}...")
    try:
        api.kernel_push(str(kernel_dir))
        print(f"  [SUCCESS] Pushed {worker_id} to Kaggle!")
    except Exception as e:
        print(f"  [ERROR] Failed to push {worker_id}: {e}")

def main():
    print("=" * 80)
    print("  KAGGLE PARALLEL GPU WORKER LAUNCHER")
    print("=" * 80)
    for spec in WORKERS_SPEC:
        push_worker_kernel(spec)

if __name__ == "__main__":
    main()
