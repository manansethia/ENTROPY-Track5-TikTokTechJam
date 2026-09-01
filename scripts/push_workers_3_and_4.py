#!/usr/bin/env python3
"""
scripts/push_workers_3_and_4.py
Pushes Kaggle Worker 3 (High-Res Robustness Benchmark) and Worker 4 (Multi-Crop Experiment).
"""

import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
WORKERS_DIR = REPO_ROOT / "kaggle_workers"
WORKERS_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = [
    {
        "id": "aigc-worker-3-highres-benchmark",
        "code": """# Worker 3: High-Resolution Robustness Benchmark
import os, sys, time, json
import torch
print("=== KAGGLE WORKER 3: HIGH-RES ROBUSTNESS BENCHMARK ===")
print("CUDA Available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
os.system("pip install -q open_clip_torch torchvision timm")
print("Environment ready for High-Res Robustness benchmark.")
"""
    },
    {
        "id": "aigc-worker-4-multicrop-ablation",
        "code": """# Worker 4: Multi-Crop Resolution Ablation
import os, sys, time, json
import torch
print("=== KAGGLE WORKER 4: MULTI-CROP ABLATION EXPERIMENT ===")
print("CUDA Available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
os.system("pip install -q open_clip_torch torchvision timm pillow")
print("Environment ready for Multi-Crop ablation experiment.")
"""
    }
]

for w in WORKERS:
    wid = w["id"]
    wdir = WORKERS_DIR / wid
    wdir.mkdir(parents=True, exist_ok=True)
    
    with open(wdir / f"{wid}.py", "w") as f:
        f.write(w["code"])
        
    meta = {
        "id": f"doubleggunther/{wid}",
        "title": wid,
        "code_file": f"{wid}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true"
    }
    with open(wdir / "kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    os.system(f"export KAGGLE_API_TOKEN='KGAT_5be018cf2a3dae2318f6980b6ce0631e' && /home/manan/.venvs/aigc-detector/bin/kaggle kernels push -p {wdir}")
