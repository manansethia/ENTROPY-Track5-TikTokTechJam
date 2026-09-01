#!/usr/bin/env python3
"""
inventory_v4_3_broad_pools.py
-----------------------------
Comprehensive inventory of all broad image pools available on Buildabot storage
for the large-scale V4.3 dataset (Target: 50,000–100,000+ unique base images).

Inventories:
  1. Real Photographic Corpus across domains (DSLR, smartphone, portrait, landscape, architecture, archival)
  2. Hard-Real Negative Pools (JPEG Q40-95, WebP, CLAHE, HDR, bilateral filtering, Lightroom edits)
  3. Full-AIGC Generator Families (SD1.5, SD2.1, SDXL, SD3, Flux, Midjourney, DALL-E, StyleGAN, ProGAN)
  4. Partial-AI Inpainting & Local Editing Pool
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = "/mnt/ai-storage/aigc_data/datasets"
REPORT_PATH = "/home/manan/aigc_robust_detection/reports/v4_3_dataset_inventory_report.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def count_images_in_dir(dpath: str) -> int:
    if not os.path.exists(dpath): return 0
    cnt = 0
    for root, _, files in os.walk(dpath):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                cnt += 1
    return cnt

def run_inventory():
    print("=" * 85)
    print("  V4.3 LARGE-SCALE DATASET INVENTORY ACROSS STORAGE")
    print("=" * 85)
    
    inventory = {
        "real_photography_pools": {},
        "hard_real_negative_pools": {},
        "full_aigc_pools": {},
        "partial_ai_pools": {},
        "total_summary": {}
    }
    
    # 1. Real Pools
    real_targets = [
        ("massive_balanced_50k/real", "Massive Balanced 50k Real DSLR/Portraits"),
        ("scaled_45k/real", "Scaled 45k Real Mixed Photography"),
        ("hires_50k_benchmark/real", "High-Resolution 50k Real"),
        ("ultra_highres_gigapixel_pool/real", "Ultra High-Res Gigapixel Real"),
        ("ntire_2026_robust_train/real", "NTIRE 2026 Real Photography"),
        ("portrait_remediation", "Studio & Social Media Real Portraits"),
        ("archival_photography_negatives", "Archival & Historical Real Photography"),
        ("vintage_archival_photos", "Vintage & Scanned Real Photography")
    ]
    
    total_real = 0
    for rel_path, desc in real_targets:
        full_p = os.path.join(BASE_DIR, rel_path)
        cnt = count_images_in_dir(full_p)
        inventory["real_photography_pools"][desc] = {"path": full_p, "count": cnt}
        total_real += cnt
        print(f"  [Real] {desc:45s}: {cnt:6,d} images")

    # 2. Hard-Real Negative Pools
    hard_targets = [
        ("hard_negative_remediation", "Post-Processed & Social Media Hard Negatives"),
        ("artbench_hard_negatives", "Complex Art & Illustration Hard Negatives"),
        ("remediation_expansion_pool", "Compression & Filtering Remediation Pool")
    ]
    
    total_hard_real = 0
    for rel_path, desc in hard_targets:
        full_p = os.path.join(BASE_DIR, rel_path)
        cnt = count_images_in_dir(full_p)
        inventory["hard_real_negative_pools"][desc] = {"path": full_p, "count": cnt}
        total_hard_real += cnt
        print(f"  [Hard-Real] {desc:40s}: {cnt:6,d} images")

    # 3. Full-AIGC Pools
    aigc_targets = [
        ("massive_balanced_50k/aigc", "Massive Balanced 50k AIGC"),
        ("scaled_45k/aigc", "Scaled 45k AIGC Multimodal"),
        ("flux_sd3_genimagepp", "Flux.1, SD3, GenImage++ State-of-the-Art"),
        ("synthbuster", "SynthBuster Multi-Generator Suite"),
        ("ultra_highres_gigapixel_pool/aigc", "Ultra High-Res AIGC Generations"),
        ("ntire_2026_robust_train/aigc", "NTIRE 2026 Robust Challenge AIGC")
    ]
    
    total_aigc = 0
    for rel_path, desc in aigc_targets:
        full_p = os.path.join(BASE_DIR, rel_path)
        cnt = count_images_in_dir(full_p)
        inventory["full_aigc_pools"][desc] = {"path": full_p, "count": cnt}
        total_aigc += cnt
        print(f"  [Full-AIGC] {desc:40s}: {cnt:6,d} images")

    # 4. Partial-AI Pools
    partial_targets = [
        ("v4_partial_ai_corpus", "V4.2 Prototype Partial-AI Corpus"),
        ("defactify", "DeFactify Forensic Splicing & Inpainting")
    ]
    
    total_partial = 0
    for rel_path, desc in partial_targets:
        full_p = os.path.join(BASE_DIR, rel_path)
        cnt = count_images_in_dir(full_p)
        inventory["partial_ai_pools"][desc] = {"path": full_p, "count": cnt}
        total_partial += cnt
        print(f"  [Partial-AI] {desc:39s}: {cnt:6,d} images")

    grand_total = total_real + total_hard_real + total_aigc + total_partial
    inventory["total_summary"] = {
        "total_real": total_real,
        "total_hard_real": total_hard_real,
        "total_aigc": total_aigc,
        "total_partial_ai": total_partial,
        "grand_total_available": grand_total
    }
    
    print("-" * 85)
    print(f"  TOTAL AUDITED IMAGES ON STORAGE: {grand_total:,d}")
    print(f"    - Pure Real       : {total_real:,d}")
    print(f"    - Hard-Real       : {total_hard_real:,d}")
    print(f"    - Full-AIGC       : {total_aigc:,d}")
    print(f"    - Partial-AI (Pre): {total_partial:,d}")
    print("=" * 85)
    
    with open(REPORT_PATH, "w") as f:
        json.dump(inventory, f, indent=2)

if __name__ == "__main__":
    run_inventory()
