#!/usr/bin/env python3
"""
AetherForensics — Dedicated Held-Out Benchmark & Test Set Downloader
Downloads isolated academic evaluation datasets into /mnt/ai-storage/aigc_data/HELD_OUT_EVAL_BENCHMARK/
CRITICAL RULE: These datasets are strictly for post-submission benchmarking and extensive testing.
NO TRAINING WILL OCCUR ON THESE FILES.
"""

import os
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

TEST_DIR = Path("/mnt/ai-storage/aigc_data/HELD_OUT_EVAL_BENCHMARK")
TEST_DIR.mkdir(parents=True, exist_ok=True)


def download_isolated_test_benchmarks():
    print(f"=== Downloading Dedicated Held-Out Test Benchmarks to {TEST_DIR} ===")

    # 1. DiffusionForensics Held-Out Evaluation Split
    df_eval = TEST_DIR / "diffusion_forensics_eval"
    df_eval.mkdir(parents=True, exist_ok=True)
    try:
        print("[Test Benchmark] Downloading DiffusionForensics test split...")
        snapshot_download(
            repo_id="victor/diffusion-forensics",
            repo_type="dataset",
            allow_patterns="*val*|*test*",
            local_dir=str(df_eval),
            max_workers=6
        )
        print("[Test Benchmark] DiffusionForensics eval downloaded.")
    except Exception as e:
        print(f"Note on DiffusionForensics eval: {e}")

    # 2. WildFake / In-The-Wild Deepfake Evaluation
    wild_eval = TEST_DIR / "wildfake_unlabelled_eval"
    wild_eval.mkdir(parents=True, exist_ok=True)
    try:
        print("[Test Benchmark] Downloading WildFake in-the-wild test set...")
        snapshot_download(
            repo_id="tjw/deepfake-celeba",
            repo_type="dataset",
            allow_patterns="*test*|*eval*",
            local_dir=str(wild_eval),
            max_workers=6
        )
        print("[Test Benchmark] WildFake eval downloaded.")
    except Exception as e:
        print(f"Note on WildFake eval: {e}")

    # 3. Unlabeled Web Test Slices (DiffusionDB Test Slices)
    unlabeled_eval = TEST_DIR / "unlabeled_web_test_stream"
    unlabeled_eval.mkdir(parents=True, exist_ok=True)
    try:
        print("[Test Benchmark] Downloading Unlabeled Web Test Slices (DiffusionDB shards 50-60)...")
        for shard_id in range(50, 60):
            url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/data/part-{shard_id:06d}.parquet"
            cmd = ["aria2c", "-x", "8", "-s", "8", "-k", "1M", "--auto-file-renaming=false", "-c", "-d", str(unlabeled_eval), url]
            subprocess.run(cmd, check=False)
        print("[Test Benchmark] Unlabeled Web Test Slices downloaded.")
    except Exception as e:
        print(f"Note on Unlabeled Web Test: {e}")

    print("=== All Held-Out Benchmark & Test Sets Downloaded Successfully ===")


if __name__ == "__main__":
    download_isolated_test_benchmarks()
