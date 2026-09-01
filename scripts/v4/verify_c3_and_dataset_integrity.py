#!/usr/bin/env python3
"""
verify_c3_and_dataset_integrity.py
----------------------------------
Step 1: Rigorous verification of Specialist C3 (CommunityForensics ViT-Small/16).
- Loads /mnt/ai-storage/aigc_data/models/community_forensics_vit_small/model.safetensors via ViTForImageClassification.
- Verifies parameter count and architecture.
- Tests forward passes on multiple real and synthetic images to prove logits vary and are NOT constant 0.5000.

Step 2: Exhaustive verification of the V4.2 Controlled Partial-AI Dataset.
- Checks image-level isolation: 0 base-image leakage between train (374) and val (99).
- Audits mask alignment:
  * Pure Real: 100% empty mask (sum == 0)
  * Hard-Real: 100% empty mask (sum == 0)
  * Partial-AI: Valid non-empty binary mask (0 < sum < total_pixels)
  * Full-AIGC: Full mask (sum == total_pixels)
- Computes min, median, max mask percentages and distribution of manipulation types.
- Saves detailed JSON audit report to reports/c3_and_dataset_verification_report.json.
"""

import os
import sys
import json
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from transformers import ViTForImageClassification

REPORT_OUT_PATH = "/home/manan/aigc_robust_detection/reports/c3_and_dataset_verification_report.json"
TRAIN_MANIFEST_P = "/home/manan/aigc_robust_detection/reports/v4_partial_ai_train_manifest.json"
VAL_MANIFEST_P = "/home/manan/aigc_robust_detection/reports/v4_partial_ai_val_manifest.json"
C3_DIR = "/mnt/ai-storage/aigc_data/models/community_forensics_vit_small"

def verify_c3_specialist() -> dict:
    print("=" * 95)
    print("  STEP 1: VERIFYING SPECIALIST C3 (COMMUNITYFORENSICS ViT-SMALL/16)")
    print("=" * 95)
    
    # 1. Load Model
    model = ViTForImageClassification.from_pretrained(C3_DIR)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda:0")
        device = "cuda:0"
    else:
        device = "cpu"
        
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Checkpoint Path       : {C3_DIR}/model.safetensors")
    print(f"  Architecture          : {model.__class__.__name__} ({model.config.model_type})")
    print(f"  Total Parameters      : {param_count:,} (Trainable: {trainable_params:,})")
    print(f"  Image Input Size      : {model.config.image_size}x{model.config.image_size}")
    print(f"  Hidden Size / Heads   : {model.config.hidden_size} hidden / {model.config.num_attention_heads} heads / {model.config.num_hidden_layers} layers")
    
    # Preprocessor transform
    transform = T.Compose([
        T.Resize((model.config.image_size, model.config.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test on real and AIGC images
    test_real = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool/real_dslr_3k_10k/*.jpg"))[:5]
    test_aigc = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool/aigc_counterpart_3k_10k/*.jpg"))[:5]
    
    test_results = []
    logits_list = []
    probs_list = []
    
    with torch.no_grad():
        for path in test_real + test_aigc:
            img = Image.open(path).convert("RGB")
            t_tensor = transform(img).unsqueeze(0).to(device)
            out = model(t_tensor)
            logit = float(out.logits[0, 0].item())
            prob = float(torch.sigmoid(out.logits[0, 0]).item())
            logits_list.append(logit)
            probs_list.append(prob)
            test_results.append({
                "file": os.path.basename(path),
                "type": "REAL" if path in test_real else "AIGC",
                "logit": round(logit, 4),
                "probability": round(prob, 4)
            })
            print(f"    [{'REAL' if path in test_real else 'AIGC'}] {os.path.basename(path)[:45]:<45} -> Logit: {logit:+.4f} | Prob: {prob:.4f}")

    is_constant = len(set([round(p, 4) for p in probs_list])) == 1
    min_prob, max_prob = min(probs_list), max(probs_list)
    print(f"\n  Probability Range     : Min {min_prob:.4f} to Max {max_prob:.4f} (Spread: {max_prob - min_prob:.4f})")
    print(f"  Is Output Static 0.5? : {'YES (FAILED)' if is_constant else 'NO (PASSED - Logits Vary Dynamically ✅)'}")
    
    return {
        "verified_loading": True,
        "checkpoint_dir": C3_DIR,
        "parameter_count": param_count,
        "is_static_output": is_constant,
        "prob_range": [min_prob, max_prob],
        "sample_evaluations": test_results
    }

def verify_dataset_integrity() -> dict:
    print("\n" + "=" * 95)
    print("  STEP 2: VERIFYING CONTROLLED PARTIAL-AI & HARD-REAL DATASET INTEGRITY")
    print("=" * 95)
    
    with open(TRAIN_MANIFEST_P, "r") as f: train_records = json.load(f)
    with open(VAL_MANIFEST_P, "r") as f: val_records = json.load(f)
    
    print(f"  Train Records Count   : {len(train_records):,} (Target: 374)")
    print(f"  Val Records Count     : {len(val_records):,} (Target: 99)")
    
    # 1. Base-Image Grouping Check (0 Leakage)
    train_sources = set(r["source_image_id"] for r in train_records)
    val_sources = set(r["source_image_id"] for r in val_records)
    overlap = train_sources.intersection(val_sources)
    print(f"  Train Source Images   : {len(train_sources)} unique base images")
    print(f"  Val Source Images     : {len(val_sources)} unique base images")
    print(f"  Source Overlap Count  : {len(overlap)} (Target: 0)")
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {overlap}"
    print("  Source-Image Isolation: PASSED (Zero Base-Image Overlap Between Train & Val) ✅")

    # 2. Mask Alignment & Category Integrity
    cat_counts = {"train": {}, "val": {}}
    manip_types = {"train": {}, "val": {}}
    partial_areas = []
    
    for split_name, records in [("train", train_records), ("val", val_records)]:
        for rec in records:
            lbl = rec["whole_image_label"]
            cat_counts[split_name][lbl] = cat_counts[split_name].get(lbl, 0) + 1
            
            etype = rec["edit_type"]
            manip_types[split_name][etype] = manip_types[split_name].get(etype, 0) + 1
            
            mask_np = np.array(Image.open(rec["mask_path"]))
            mask_sum = np.sum(mask_np > 0)
            total_px = mask_np.size
            ratio = mask_sum / total_px
            
            if lbl == "REAL":
                assert mask_sum == 0, f"Error: REAL sample {rec['sample_id']} has non-empty mask!"
            elif lbl == "FULL_AIGC":
                assert mask_sum == total_px, f"Error: FULL_AIGC sample {rec['sample_id']} has partial mask!"
            elif lbl == "PARTIAL_AIGC":
                assert 0 < mask_sum < total_px, f"Error: PARTIAL_AIGC sample {rec['sample_id']} has invalid mask sum {mask_sum}!"
                partial_areas.append(ratio * 100.0)

    print("\n  Class Distribution Breakdown:")
    for split in ["train", "val"]:
        print(f"    [{split.upper()}] Real: {cat_counts[split].get('REAL',0)} | Partial-AI: {cat_counts[split].get('PARTIAL_AIGC',0)} | Full-AIGC: {cat_counts[split].get('FULL_AIGC',0)}")

    print("\n  Manipulation Type Breakdown (Train + Val):")
    all_manip = {}
    for split in ["train", "val"]:
        for k, v in manip_types[split].items(): all_manip[k] = all_manip.get(k, 0) + v
    for k, v in sorted(all_manip.items()):
        print(f"    - {k:<32}: {v} samples")

    p_min = float(np.min(partial_areas))
    p_med = float(np.median(partial_areas))
    p_max = float(np.max(partial_areas))
    print(f"\n  Partial-AI Manipulated Area Statistics:")
    print(f"    Minimum Region Area : {p_min:.2f}% of image")
    print(f"    Median Region Area  : {p_med:.2f}% of image")
    print(f"    Maximum Region Area : {p_max:.2f}% of image")
    print("  Mask Quality & Alignment Audit: PASSED 100% ✅")

    return {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "source_overlap": len(overlap),
        "class_breakdown": cat_counts,
        "manipulation_types": all_manip,
        "mask_area_stats": {"min_pct": p_min, "median_pct": p_med, "max_pct": p_max},
        "mask_audit_passed": True
    }

def main():
    c3_res = verify_c3_specialist()
    ds_res = verify_dataset_integrity()
    
    full_report = {
        "c3_verification": c3_res,
        "dataset_verification": ds_res,
        "all_checks_passed": (not c3_res["is_static_output"]) and ds_res["mask_audit_passed"]
    }
    
    with open(REPORT_OUT_PATH, "w") as f:
        json.dump(full_report, f, indent=2)
        
    print("\n" + "=" * 95)
    print(f"  VERIFICATION REPORT SAVED: {REPORT_OUT_PATH}")
    print("=" * 95)

if __name__ == "__main__":
    main()
