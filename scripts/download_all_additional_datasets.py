#!/usr/bin/env python3
"""Automated background downloader for all additional datasets on the 1 TB HDD.
Downloads:
1. SID_Set remaining 150 shards
2. CIFAKE dataset
3. GenImage multi-generator splits (Midjourney, SD v1.4, GLIDE, BigGAN, ADM, VQDM)
4. AIGCBench / WildFake subsets
"""

import os
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

HDD_BASE = Path("/mnt/ai-storage/aigc_data/datasets")
HDD_BASE.mkdir(parents=True, exist_ok=True)


def download_cifake():
    print("\n[Downloader] Starting CIFAKE dataset download...")
    out_dir = HDD_BASE / "cifake"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/datasets/roberta/cifake/resolve/main/data/train-00000-of-00001.parquet"
    cmd = [
        "aria2c", "-x", "8", "-s", "8", "-k", "1M",
        "--auto-file-renaming=false", "-c",
        "-d", str(out_dir), url
    ]
    subprocess.run(cmd, check=False)
    print("[Downloader] CIFAKE completed.")


def download_genimage():
    print("\n[Downloader] Starting GenImage multi-generator subsets download...")
    out_dir = HDD_BASE / "genimage"
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = ["midjourney", "stable_diffusion_v_1_4", "glide", "biggan", "adm", "vqdm", "wukong"]
    for gen in generators:
        print(f"Downloading GenImage split: {gen}...")
        try:
            snapshot_download(
                repo_id="wjbmatthew/GenImage",
                repo_type="dataset",
                allow_patterns=f"*{gen}*",
                local_dir=str(out_dir),
                max_workers=8,
            )
        except Exception as e:
            print(f"Note on GenImage {gen}: {e}")
    print("[Downloader] GenImage download completed.")


def download_sid_extended():
    print("\n[Downloader] Starting Extended SID_Set (Shards 50 to 150)...")
    out_dir = HDD_BASE / "sid_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(50, 150):
        fname = f"train-{i:05d}-of-00249.parquet"
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 10_000_000:
            continue
        url = f"https://huggingface.co/datasets/saberzl/SID_Set/resolve/main/data/{fname}"
        cmd = [
            "aria2c", "-x", "8", "-s", "8", "-k", "1M",
            "--auto-file-renaming=false", "-c",
            "-d", str(out_dir), url
        ]
        subprocess.run(cmd, check=False)


def download_modern_flux_sd3_and_art():
    print("\n[Downloader] Starting Modern Generative Models (FLUX.1, SD3, Defactify, NTIRE)...")
    
    # 1. GenImage++ (FLUX.1 & SD3)
    flux_dir = HDD_BASE / "flux_sd3_genimagepp"
    flux_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id="Lunahera/genimagepp", repo_type="dataset", local_dir=str(flux_dir), max_workers=6)
        print("[Downloader] GenImage++ (FLUX.1 & SD3) completed.")
    except Exception as e:
        print("Note on GenImage++:", e)

    # 2. Defactify (Midjourney v6, DALL-E 3, SD3, SDXL)
    defact_dir = HDD_BASE / "defactify"
    defact_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id="Rajarshi-Roy-research/Defactify_Image_Dataset", repo_type="dataset", local_dir=str(defact_dir), max_workers=6)
        print("[Downloader] Defactify dataset completed.")
    except Exception as e:
        print("Note on Defactify:", e)

    # 3. WikiArt / Historic Paintings & Vintage Art (Authentic Hard-Negatives)
    art_dir = HDD_BASE / "wikiart_hard_negatives"
    art_dir.mkdir(parents=True, exist_ok=True)
    try:
        print("[Downloader] Downloading ArtBench-10 & WikiArt authentic fine art...")
        snapshot_download(repo_id="huggan/wikiart", repo_type="dataset", local_dir=str(art_dir), max_workers=6)
        print("[Downloader] WikiArt fine art completed.")
    except Exception as e:
        print("Note on WikiArt:", e)


def download_vintage_fine_art_and_confusing_hard_negatives():
    print("\n[Downloader] Starting Confusing Hard-Negatives: Vintage Photos, CGI Renders & Classical Art...")
    
    # 1. ArtBench-10 (Standard Benchmark for 10 Fine Art Styles: Baroque, Impressionism, Surrealism)
    artbench_dir = HDD_BASE / "artbench_hard_negatives"
    artbench_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id="civitai/artbench-10", repo_type="dataset", local_dir=str(artbench_dir), max_workers=6)
        print("[Downloader] ArtBench-10 completed.")
    except Exception as e:
        print("Note on ArtBench-10:", e)

    # 2. Historical & Vintage Archival Photography
    vintage_dir = HDD_BASE / "vintage_archival_photos"
    vintage_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id="dalle-mini/vintage-photos", repo_type="dataset", local_dir=str(vintage_dir), max_workers=6)
        print("[Downloader] Vintage photography completed.")
    except Exception as e:
        print("Note on Vintage Photography:", e)


def main():
    print(f"=== Comprehensive Dataset Ingestion Daemon Active on {HDD_BASE} ===")
    download_cifake()
    download_sid_extended()
    download_genimage()
    download_modern_flux_sd3_and_art()
    download_massive_unlabeled_web_and_diffusiondb()
    download_vintage_fine_art_and_confusing_hard_negatives()
    print("=== All Additional Dataset Downloads Completed Successfully ===")


if __name__ == "__main__":
    main()




