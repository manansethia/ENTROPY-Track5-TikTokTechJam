#!/usr/bin/env python3
"""Download high-priority targeted forensic datasets & benchmark models:
1. AIGI Quality Paradox (~7.5 GB) - High-quality hard AI & metadata pairs
2. DDA (Dual Data Alignment) SOTA Baseline Checkpoint
3. Synthbuster (Diverse cross-generator benchmark)
4. AIGIBench evaluation subsets
"""

import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

DATASETS_DIR = Path("/mnt/ai-storage/aigc_data/datasets")
MODELS_DIR = Path("/mnt/ai-storage/aigc_data/models")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_quality_paradox():
    print("\n[Priority 1] Starting AIGI-Detection-Quality-Paradox (7.5 GB)...")
    dest = DATASETS_DIR / "aigi_quality_paradox"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="Coxy7/AIGI-Detection-Quality-Paradox",
            repo_type="dataset",
            local_dir=str(dest),
            max_workers=6,
        )
        print("--> AIGI-Detection-Quality-Paradox downloaded successfully!")
    except Exception as e:
        print(f"Error downloading Quality Paradox: {e}")


def download_dda_model():
    print("\n[Priority 2] Starting DDA (Dual-Data-Alignment) SOTA Checkpoint...")
    dest = MODELS_DIR / "dda_dual_data_alignment"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="Junwei-Xi/Dual-Data-Alignment",
            local_dir=str(dest),
            max_workers=4,
        )
        print("--> DDA model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading DDA model: {e}")


def download_synthbuster():
    print("\n[Priority 3] Starting Synthbuster Dataset (Zenodo)...")
    dest = DATASETS_DIR / "synthbuster"
    dest.mkdir(parents=True, exist_ok=True)
    
    # Download Synthbuster zip using aria2c
    url = "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
    zip_file = dest / "synthbuster.zip"
    if not (dest / "dalle2").exists():
        cmd = ["aria2c", "-x", "8", "-s", "8", "-k", "1M", "-o", "synthbuster.zip", "-d", str(dest), url]
        print("Downloading Synthbuster zip...")
        subprocess.run(cmd, check=False)
        if zip_file.exists():
            print("Extracting Synthbuster zip...")
            subprocess.run(["unzip", "-q", "-o", str(zip_file), "-d", str(dest)], check=False)
            print("--> Synthbuster extracted successfully!")
    else:
        print("--> Synthbuster already present!")


def download_aigibench():
    print("\n[Priority 4] Starting AIGIBench Evaluation Benchmark (HorizonTEL/AIGIBench)...")
    dest = DATASETS_DIR / "aigibench_eval"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="HorizonTEL/AIGIBench",
            repo_type="dataset",
            local_dir=str(dest),
            max_workers=6,
        )
        print("--> AIGIBench evaluation suite downloaded successfully!")
    except Exception as e:
        print(f"Error downloading AIGIBench: {e}")


def main():
    print("=== Targeted Forensic Dataset Acquisition Pipeline Active ===")
    download_quality_paradox()
    download_dda_model()
    download_synthbuster()
    download_aigibench()
    print("\n=== Targeted Forensic Ingestion Completed ===")


if __name__ == "__main__":
    main()
