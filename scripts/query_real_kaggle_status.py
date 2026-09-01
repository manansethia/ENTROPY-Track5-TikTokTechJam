#!/usr/bin/env python3
"""
scripts/query_real_kaggle_status.py
Dynamically queries the Kaggle API for real kernels, datasets, and account status.
No hardcoded strings.
"""

import os
import sys
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

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    username = os.environ.get("KAGGLE_USERNAME", "Unknown")
    print(f"AUTHENTICATED_USERNAME: {username}")
    
    # Query real kernels
    kernels = api.kernels_list(mine=True)
    print(f"REAL_KERNELS_COUNT: {len(kernels)}")
    for k in kernels:
        print(f"  Kernel: {getattr(k, 'ref', str(k))} | Last run: {getattr(k, 'lastRunTime', 'N/A')}")
        
    # Query datasets
    datasets = api.dataset_list(user=username)
    print(f"REAL_DATASETS_COUNT: {len(datasets)}")
    for d in datasets:
        print(f"  Dataset: {getattr(d, 'ref', str(d))}")
        
except Exception as e:
    print(f"KAGGLE_ERROR: {e}")
