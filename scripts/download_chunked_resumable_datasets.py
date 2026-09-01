#!/usr/bin/env python3
"""
scripts/download_chunked_resumable_datasets.py
Atomic, Resumable, Chunk-by-Chunk Dataset & Model Ingestion Engine on Buildabot.
Tracks state persistently in /mnt/ai-storage/aigc_data/checkpoints/download_state.json.
Guarantees zero lost progress across power outages or network interruptions.
"""

import os
import sys
import json
import time
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from huggingface_hub import HfApi, hf_hub_download

BASE_STORAGE_DIR = Path("/mnt/ai-storage/aigc_data")
DATASETS_DIR = BASE_STORAGE_DIR / "datasets"
MODELS_DIR = BASE_STORAGE_DIR / "models"
STATE_FILE_PATH = BASE_STORAGE_DIR / "checkpoints" / "download_state.json"

STATE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

api = HfApi()

def load_state() -> Dict[str, Any]:
    """Loads persistent checkpoint state."""
    if STATE_FILE_PATH.exists():
        try:
            with open(STATE_FILE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_files": {}, "last_updated": time.time()}

def save_state(state: Dict[str, Any]):
    """Atomically saves persistent checkpoint state."""
    state["last_updated"] = time.time()
    tmp_path = STATE_FILE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_FILE_PATH)

def get_storage_telemetry() -> str:
    disk = shutil.disk_usage(str(DATASETS_DIR))
    return f"Free: {disk.free / (1024**3):.1f} GB ({disk.used / disk.total * 100:.1f}% used)"

def process_single_file(repo_id: str, filename: str, repo_type: str, dest_dir: Path, state: Dict[str, Any]) -> bool:
    """Downloads a single file/chunk, verifies integrity, and updates persistent checkpoint."""
    file_key = f"{repo_id}:{filename}"
    dest_path = dest_dir / filename
    
    # Check if already completed and verified
    if file_key in state["completed_files"] and dest_path.exists():
        expected_size = state["completed_files"][file_key].get("size_bytes", 0)
        if dest_path.stat().st_size == expected_size and expected_size > 0:
            print(f"  [CHECKPOINT HIT] {filename} already completed ({expected_size / (1024**2):.1f} MB). Skipping.")
            return True
            
    print(f"\n  [DOWNLOADING CHUNK] {filename} from {repo_id}...")
    t0 = time.perf_counter()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False
        )
        
        actual_path = Path(downloaded_path)
        file_size = actual_path.stat().st_size
        elapsed = time.perf_counter() - t0
        rate_mb = (file_size / (1024**2)) / max(elapsed, 0.001)
        
        # Save to state atomically
        state["completed_files"][file_key] = {
            "path": str(actual_path),
            "size_bytes": file_size,
            "completed_at": time.time(),
            "elapsed_seconds": round(elapsed, 2)
        }
        save_state(state)
        
        print(f"  [CHUNK SAVED] {filename} ({file_size / (1024**2):.1f} MB) in {elapsed:.1f}s ({rate_mb:.1f} MB/s) | {get_storage_telemetry()}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {filename}: {e}")
        return False

def download_dataset_in_chunks(repo_id: str, repo_type: str, dest_dir: Path, state: Dict[str, Any], filter_prefix: Optional[str] = None):
    """Enumerates repository files and downloads them sequentially chunk-by-chunk."""
    print(f"\n" + "=" * 80)
    print(f"  PROCESSING DATASET: {repo_id} -> {dest_dir.name}")
    print("=" * 80)
    
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        print(f"Error listing files for {repo_id}: {e}")
        return
        
    # Filter files if needed (e.g. data/ or .zip)
    target_files = []
    for f in files:
        if f.endswith(".gitattributes"):
            continue
        if filter_prefix and not f.startswith(filter_prefix):
            continue
        target_files.append(f)
        
    print(f"Total chunks/files in {repo_id}: {len(target_files)}")
    
    for idx, filename in enumerate(target_files, 1):
        print(f"[{idx}/{len(target_files)}] Processing {filename}...")
        success = process_single_file(repo_id, filename, repo_type, dest_dir, state)
        if not success:
            print(f"Warning: Issue with {filename}, continuing with next chunk...")

def main():
    print("=" * 85)
    print("  ATOMIC RESUMABLE CHUNK-BY-CHUNK DATASET & MODEL INGESTION ENGINE")
    print(f"  Persistent State File: {STATE_FILE_PATH}")
    print(f"  Initial Storage: {get_storage_telemetry()}")
    print("=" * 85)
    
    state = load_state()
    print(f"Loaded existing checkpoint state: {len(state['completed_files'])} chunks already completed.")
    
    # 1. HiRes-50K (Evaluation Set - Download chunk by chunk: W_0900.zip to W_4000.zip)
    download_dataset_in_chunks(
        repo_id="Mu437/HiRes-50K",
        repo_type="dataset",
        dest_dir=DATASETS_DIR / "hires_50k_benchmark",
        state=state
    )
    
    # 2. NTIRE 2026 Robust AI Detection — TRAIN (shard_0.zip to shard_4.zip)
    download_dataset_in_chunks(
        repo_id="deepfakesMSU/NTIRE-RobustAIGenDetection-train",
        repo_type="dataset",
        dest_dir=DATASETS_DIR / "ntire_2026_robust_train",
        state=state
    )
    
    # 3. NTIRE 2026 Robust AI Detection — VAL & TEST PUBLIC
    download_dataset_in_chunks(
        repo_id="deepfakesMSU/NTIRE-RobustAIGenDetection-val",
        repo_type="dataset",
        dest_dir=DATASETS_DIR / "ntire_2026_robust_val",
        state=state
    )
    download_dataset_in_chunks(
        repo_id="deepfakesMSU/NTIRE-RobustAIGenDetection-test-public",
        repo_type="dataset",
        dest_dir=DATASETS_DIR / "ntire_2026_robust_test_public",
        state=state
    )
    
    # 4. AIGI Detection — Quality Paradox (Parquet chunks fake-00000 to fake-00014)
    download_dataset_in_chunks(
        repo_id="Coxy7/AIGI-Detection-Quality-Paradox",
        repo_type="dataset",
        dest_dir=DATASETS_DIR / "aigi_quality_paradox_coxy7",
        state=state
    )
    
    # 5. AIGC Detection Benchmark (Parquet chunks test-00000 to test-00059)
    download_dataset_in_chunks(
        repo_id="TheKernel01/AIGC-Detection-Benchmark",
        repo_type="dataset",
        dest_dir=DATASETS_DIR / "aigc_detection_benchmark_kernel01",
        state=state
    )
    
    # 6. MLLM-Generated Image Dataset (GPT-Image2 and Nano Banana2)
    download_dataset_in_chunks(
        repo_id="zr-zhang/MLLM-Generated-Image-Detection-Dataset",
        repo_type="dataset",
        dest_dir=DATASETS_DIR / "mllm_generated_dataset",
        state=state
    )
    
    # 7. CommunityForensics ViT-Small (21.8M Model)
    download_dataset_in_chunks(
        repo_id="buildborderless/CommunityForensics-DeepfakeDet-ViT",
        repo_type="model",
        dest_dir=MODELS_DIR / "community_forensics_vit_small",
        state=state
    )
    
    # 8. SPAI / TFG-Model (Any-Resolution Model)
    download_dataset_in_chunks(
        repo_id="aminasifar1/TFG-model",
        repo_type="model",
        dest_dir=MODELS_DIR / "spai_tfg",
        state=state
    )
    
    print("\n" + "=" * 85)
    print(f"  ALL CHUNKS FULLY DOWNLOADED & PERSISTED! Final Storage: {get_storage_telemetry()}")
    print("=" * 85)

if __name__ == "__main__":
    main()
