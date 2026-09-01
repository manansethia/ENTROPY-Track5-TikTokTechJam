#!/usr/bin/env python3
"""
scripts/download_all_requested_datasets.py
High-Speed Master Parallel Downloader for all user-requested datasets and models on Buildabot:
1. NTIRE 2026 Robust TRAIN (deepfakesMSU/NTIRE-RobustAIGenDetection-train)
2. NTIRE 2026 Robust VAL (deepfakesMSU/NTIRE-RobustAIGenDetection-val)
3. NTIRE 2026 Robust TEST-PUBLIC (deepfakesMSU/NTIRE-RobustAIGenDetection-test-public)
4. HiRes-50K Evaluation Set (Mu437/HiRes-50K) [EVAL ONLY]
5. AIGC Detection Benchmark (TheKernel01/AIGC-Detection-Benchmark)
6. AIGI Detection Quality Paradox (Coxy7/AIGI-Detection-Quality-Paradox)
7. MLLM-Generated Dataset (zr-zhang/MLLM-Generated-Image-Detection-Dataset)
8. CommunityForensics ViT-Small (buildborderless/CommunityForensics-DeepfakeDet-ViT)
9. SPAI / TFG Model (aminasifar1/TFG-model)
"""

import os
import sys
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import snapshot_download, HfApi
import psutil

TARGET_DATASETS_DIR = Path("/mnt/ai-storage/aigc_data/datasets")
TARGET_MODELS_DIR = Path("/mnt/ai-storage/aigc_data/models")

TARGET_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
TARGET_MODELS_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_TARGETS = [
    {
        "name": "HiRes_50K_Eval_Benchmark",
        "repo_id": "Mu437/HiRes-50K",
        "repo_type": "dataset",
        "local_dir": TARGET_DATASETS_DIR / "hires_50k_benchmark",
        "note": "STRICTLY EVALUATION ONLY - DO NOT TRAIN"
    },
    {
        "name": "NTIRE_2026_Robust_TRAIN",
        "repo_id": "deepfakesMSU/NTIRE-RobustAIGenDetection-train",
        "repo_type": "dataset",
        "local_dir": TARGET_DATASETS_DIR / "ntire_2026_robust_train",
        "note": "TRAINING DATASET - 277K images"
    },
    {
        "name": "NTIRE_2026_Robust_VAL",
        "repo_id": "deepfakesMSU/NTIRE-RobustAIGenDetection-val",
        "repo_type": "dataset",
        "local_dir": TARGET_DATASETS_DIR / "ntire_2026_robust_val",
        "note": "VALIDATION DATASET"
    },
    {
        "name": "NTIRE_2026_Robust_TEST_PUBLIC",
        "repo_id": "deepfakesMSU/NTIRE-RobustAIGenDetection-test-public",
        "repo_type": "dataset",
        "local_dir": TARGET_DATASETS_DIR / "ntire_2026_robust_test_public",
        "note": "PUBLIC TEST BENCHMARK"
    },
    {
        "name": "AIGC_Detection_Benchmark",
        "repo_id": "TheKernel01/AIGC-Detection-Benchmark",
        "repo_type": "dataset",
        "local_dir": TARGET_DATASETS_DIR / "aigc_detection_benchmark_kernel01",
        "note": "EVALUATION & GENERATOR BENCHMARK - 125K images"
    },
    {
        "name": "AIGI_Quality_Paradox",
        "repo_id": "Coxy7/AIGI-Detection-Quality-Paradox",
        "repo_type": "dataset",
        "local_dir": TARGET_DATASETS_DIR / "aigi_quality_paradox_coxy7",
        "note": "HARD AIGC 24K images"
    },
    {
        "name": "MLLM_Generated_Dataset",
        "repo_id": "zr-zhang/MLLM-Generated-Image-Detection-Dataset",
        "repo_type": "dataset",
        "local_dir": TARGET_DATASETS_DIR / "mllm_generated_dataset",
        "note": "GPT Image2 & Nano Banana2 - 4.3K images"
    },
    {
        "name": "CommunityForensics_ViT_Small",
        "repo_id": "buildborderless/CommunityForensics-DeepfakeDet-ViT",
        "repo_type": "model",
        "local_dir": TARGET_MODELS_DIR / "community_forensics_vit_small",
        "note": "21.8M PARAMETER DETECTOR"
    },
    {
        "name": "SPAI_TFG_Model",
        "repo_id": "aminasifar1/TFG-model",
        "repo_type": "model",
        "local_dir": TARGET_MODELS_DIR / "spai_tfg",
        "note": "ANY-RESOLUTION DETECTOR"
    }
]

def get_disk_info():
    disk = shutil.disk_usage(str(TARGET_DATASETS_DIR))
    return f"Free: {disk.free / (1024**3):.1f} GB ({disk.used / disk.total * 100:.1f}% used)"

def download_target(target):
    name = target["name"]
    repo_id = target["repo_id"]
    repo_type = target["repo_type"]
    local_dir = str(target["local_dir"])
    note = target["note"]
    
    print(f"\n[STARTING] {name} ({repo_id}) -> {local_dir} [{note}]...")
    t0 = time.perf_counter()
    try:
        res = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=local_dir,
            max_workers=8,
            resume_download=True
        )
        elapsed = time.perf_counter() - t0
        print(f"[COMPLETED] {name} in {elapsed:.1f}s | {get_disk_info()}")
        return {"name": name, "status": "SUCCESS", "elapsed": elapsed, "path": local_dir}
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return {"name": name, "status": "FAILED", "error": str(e)}

def main():
    print("=" * 80)
    print("  HIGH-SPEED MASTER PARALLEL DATASET & MODEL DOWNLOADER")
    print(f"  Target Storage: /mnt/ai-storage/aigc_data/ | Initial {get_disk_info()}")
    print("=" * 80)
    
    t_start = time.perf_counter()
    # Download concurrently across 4 dataset workers to maximize network bandwidth
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_target, t): t for t in DOWNLOAD_TARGETS}
        for f in as_completed(futures):
            res = f.result()
            print(f"  --> Result for {res['name']}: {res['status']}")
            
    total_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"  ALL DOWNLOADS COMPLETED in {total_elapsed:.1f}s! Final Storage {get_disk_info()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
