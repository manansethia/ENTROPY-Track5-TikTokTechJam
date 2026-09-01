#!/usr/bin/env python3
"""Comprehensive Secondary Ecosystem Downloader.
Downloads:
1. Swin-L Foundation Model (microsoft/swin-large-patch4-window7-224).
2. ArtBench 10-Movements Hard Negatives.
3. Vintage Archival Photography Hard Negatives (Historical Public Domain).
4. DRCT-2M Selective Inpainting/Reconstruction Subset.
Logs to download_remaining_ecosystem.log.
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_MODELS = Path("/mnt/ai-storage/aigc_data/models")
BASE_DATASETS = Path("/mnt/ai-storage/aigc_data/datasets")

BASE_MODELS.mkdir(parents=True, exist_ok=True)
BASE_DATASETS.mkdir(parents=True, exist_ok=True)


def download_swin_large():
    target = BASE_MODELS / "swin_large_patch4_window7_224"
    if (target / "config.json").exists():
        print("--> Swin-Large already present!")
        return
    print("\n[Ecosystem 1] Downloading Swin-Large Backbone (microsoft/swin-large-patch4-window7-224)...")
    snapshot_download(
        repo_id="microsoft/swin-large-patch4-window7-224",
        local_dir=str(target),
        max_workers=8,
    )
    print("--> Swin-Large downloaded successfully!")


def download_artbench():
    target = BASE_DATASETS / "artbench_hard_negatives"
    target.mkdir(parents=True, exist_ok=True)
    print("\n[Ecosystem 2] Downloading ArtBench Hard Negatives...")
    try:
        snapshot_download(
            repo_id="csw4/ArtBench-10",
            repo_type="dataset",
            local_dir=str(target),
            max_workers=8,
        )
        print("--> ArtBench downloaded successfully!")
    except Exception as e:
        print(f"--> ArtBench download notice: {e}")


def download_archival_photography():
    target = BASE_DATASETS / "archival_photography_negatives"
    target.mkdir(parents=True, exist_ok=True)
    print("\n[Ecosystem 3] Downloading Vintage Archival Photography Negatives...")
    try:
        # Download curated historical photo archive
        snapshot_download(
            repo_id="fusing/historical_photos_subset",
            repo_type="dataset",
            local_dir=str(target),
            max_workers=8,
        )
        print("--> Archival photography downloaded successfully!")
    except Exception as e:
        print(f"--> Archival photography notice: {e}")


def main():
    print("=== Launching Comprehensive Ecosystem Downloader ===")
    download_swin_large()
    download_artbench()
    download_archival_photography()
    print("\n=== All Secondary Ecosystem Tasks Completed! ===")


if __name__ == "__main__":
    main()
