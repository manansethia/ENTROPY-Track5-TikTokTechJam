#!/usr/bin/env python3
"""
scripts/launch_kaggle_t4_selftest.py
Explicitly requests NvidiaTeslaT4 accelerator (sm_75) in Kaggle metadata.
"""

import os
import sys
import json
from pathlib import Path

REPO_ROOT = Path("/home/manan/aigc_robust_detection")
WORKER_DIR = REPO_ROOT / "kaggle_workers" / "gpu-selftest"

# Metadata explicitly requesting NvidiaTeslaT4
meta = {
    "id": "doubleggunther/aigc-gpu-selftest",
    "title": "aigc-gpu-selftest",
    "code_file": "gpu-selftest.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "accelerator": "nvidiaTeslaT4"
}

with open(WORKER_DIR / "kernel-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Updated metadata with accelerator=nvidiaTeslaT4:\n{json.dumps(meta, indent=2)}")
