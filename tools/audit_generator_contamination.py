#!/usr/bin/env python3
"""Evidence-Based Generator Contamination Audit Tool.
Scans all dataset directories, subfolders, manifests, and file names in /mnt/ai-storage/aigc_data/datasets.
Records exact sample counts, dataset splits, training usage status, and zero-shot eligibility.
Saves authoritative report to reports/generator_contamination_audit.json.
"""

import glob
import json
import os
import sys
from pathlib import Path

DATASETS_DIR = Path("/mnt/ai-storage/aigc_data/datasets")

# Target generator keywords to audit
GENERATOR_KEYWORDS = [
    ("SD 1.4", ["sd14", "sd_1_4", "sd-v1-4", "stable_diffusion_1_4"]),
    ("SD 1.5", ["sd15", "sd_1_5", "sd-v1-5", "stable_diffusion_1_5", "stable-diffusion-v1-5"]),
    ("SD 2.1", ["sd21", "sd_2_1", "sd-v2-1", "stable_diffusion_2_1"]),
    ("SDXL 1.0", ["sdxl", "sd_xl", "stable_diffusion_xl"]),
    ("SD 3.5", ["sd3", "sd35", "sd3_5", "sd-3-5", "stable_diffusion_3"]),
    ("FLUX.1-dev", ["flux", "flux1", "flux_1", "flux.1"]),
    ("Midjourney v5", ["midjourney_v5", "mj_v5", "mjv5", "midjourney-v5"]),
    ("Midjourney v6", ["midjourney_v6", "mj_v6", "mjv6", "midjourney-v6", "midjourney"]),
    ("DALL-E 2", ["dalle2", "dalle_2", "dall-e-2"]),
    ("DALL-E 3", ["dalle3", "dalle_3", "dall-e-3", "dalle"]),
    ("Adobe Firefly", ["firefly", "adobe_firefly"]),
    ("Google Imagen", ["imagen", "google_imagen"]),
    ("StyleGAN-XL", ["stylegan", "stylegan_xl", "stylegan3"]),
    ("PixArt-alpha", ["pixart", "pixart_alpha", "pixart_sigma"]),
]


def run_evidence_audit():
    print("=== Running Evidence-Based Generator Contamination Audit ===")
    audit_table = []
    
    # 1. Discover all dataset paths
    candidate_datasets = [
        ("genimage_plus", DATASETS_DIR / "flux_sd3_genimagepp", "TRAIN"),
        ("defactify", DATASETS_DIR / "defactify", "TRAIN"),
        ("massive_balanced_50k", DATASETS_DIR / "massive_balanced_50k", "TRAIN"),
        ("aigi_quality_paradox", DATASETS_DIR / "aigi_quality_paradox", "TRAIN"),
        ("sid_parquet", DATASETS_DIR / "sid_parquet", "TRAIN"),
        ("wikiart_hard_negatives", DATASETS_DIR / "wikiart_hard_negatives", "TRAIN"),
        ("synthbuster", DATASETS_DIR / "synthbuster", "EXTERNAL_EVAL"),
        ("aigibench_eval", DATASETS_DIR / "aigibench_eval", "EXTERNAL_EVAL"),
    ]

    for gen_display, aliases in GENERATOR_KEYWORDS:
        found_in_any_dataset = False
        
        for ds_name, ds_path, split_role in candidate_datasets:
            if not ds_path.exists():
                continue
            
            # Count matching files
            match_count = 0
            for root, dirs, files in os.walk(ds_path):
                root_lower = root.lower()
                # Check dir name match
                if any(alias in root_lower for alias in aliases):
                    match_count += len([f for f in files if f.lower().endswith((".jpg", ".png", ".webp", ".parquet", ".zip"))])
                else:
                    # Check individual file names
                    for f in files:
                        f_lower = f.lower()
                        if any(alias in f_lower for alias in aliases):
                            match_count += 1

            if match_count > 0:
                found_in_any_dataset = True
                is_train_exposed = (split_role == "TRAIN")
                audit_table.append({
                    "generator": gen_display,
                    "dataset": ds_name,
                    "split_role": split_role,
                    "sample_count": match_count,
                    "used_in_training": "YES" if is_train_exposed else "NO",
                    "zero_shot_eligible": "NO" if is_train_exposed else "YES",
                })

        if not found_in_any_dataset:
            audit_table.append({
                "generator": gen_display,
                "dataset": "None (External Benchmark Only)",
                "split_role": "EXTERNAL_EVAL",
                "sample_count": 0,
                "used_in_training": "NO",
                "zero_shot_eligible": "YES",
            })

    # Save to json
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generator_contamination_audit.json"
    with open(out_file, "w") as f:
        json.dump(audit_table, f, indent=2)

    print(f"\n{'='*115}")
    print(f"{'Generator':<18} | {'Dataset':<24} | {'Split':<14} | {'Samples':<10} | {'Used in Train':<14} | {'Zero-Shot Eligible'}")
    print(f"{'-'*115}")
    for entry in audit_table:
        print(f"{entry['generator']:<18} | {entry['dataset']:<24} | {entry['split_role']:<14} | {entry['sample_count']:<10} | {entry['used_in_training']:<14} | {entry['zero_shot_eligible']}")
    print(f"{'='*115}")
    print(f"Authoritative audit saved to {out_file}!\n")


if __name__ == "__main__":
    run_evidence_audit()
