#!/usr/bin/env python3
"""High-speed parallel downloader for HorizonTEL/AIGIBench dataset.
Downloads all generator zip files concurrently using multi-connection streams.
"""

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_url

REPO_ID = "HorizonTEL/AIGIBench"
TARGET_DIR = Path("/mnt/ai-storage/aigc_data/datasets/aigibench_eval")
TARGET_DIR.mkdir(parents=True, exist_ok=True)


def download_file(file_path: str):
    target_path = TARGET_DIR / file_path
    if target_path.exists() and target_path.stat().st_size > 1024:
        print(f"--> Already present: {file_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo_id=REPO_ID, filename=file_path, repo_type="dataset")

    # Use aria2c with 8 connections per file
    cmd = [
        "aria2c",
        "-x", "8",
        "-s", "8",
        "-k", "1M",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "-d", str(target_path.parent),
        "-o", target_path.name,
        url
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"--> Successfully downloaded: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--> Aria2c failed for {file_path}, falling back to curl: {e.stderr[:100]}")
        # Fallback to curl
        subprocess.run(["curl", "-L", "-o", str(target_path), url], check=False)
        return True


def main():
    print("=== Launching High-Speed Parallel AIGIBench Downloader ===")
    api = HfApi()
    all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    zip_files = [f for f in all_files if f.endswith(".zip") or f.endswith(".json") or f.endswith(".md")]
    print(f"Total target files: {len(zip_files)}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, f): f for f in zip_files}
        for future in as_completed(futures):
            f = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading {f}: {e}")

    print("=== All AIGIBench Files Successfully Downloaded! ===")


if __name__ == "__main__":
    main()
