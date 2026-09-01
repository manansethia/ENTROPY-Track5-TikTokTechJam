# =====================================================================================
# MASTER HIGH-RESOLUTION REMEDIATION MANIFEST BUILDER (FULL 260K+ POOL)
# Scans 5 Extracted NTIRE Shards + DIV2K 2K/4K + Portrait Pool + Old Governed Train Base
# =====================================================================================

import os, sys, time, json, glob, hashlib
from pathlib import Path

print("=" * 85)
print("  COMPILING COMPREHENSIVE MASTER HIGH-RES MANIFEST (260K+ POOL)")
print("=" * 85)

master_samples = []
real_count = 0
aigc_count = 0
sources = {}

# 1. SCAN ALL 5 EXTRACTED NTIRE SHARDS (0, 1, 2, 3, 4)
ntire_dir = Path("/mnt/ai-storage/aigc_data/datasets/ntire_2026_robust_train/extracted")
if ntire_dir.exists():
    print("Scanning all 5 extracted NTIRE shards...")
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for p in ntire_dir.rglob(ext):
            p_str = str(p)
            lbl = 1 if any(k in p_str.lower() for k in ["aigc", "synthetic", "fake", "gen", "flux", "midjourney", "sdxl"]) else 0
            if lbl == 0:
                real_count += 1
            else:
                aigc_count += 1
            src = "NTIRE_2026_HighRes_Train"
            sources[src] = sources.get(src, 0) + 1
            master_samples.append({
                "sample_id": f"NTIRE_{len(master_samples):07d}",
                "canonical_path": p_str,
                "label": lbl,
                "source": src,
                "split": "TRAIN"
            })

# 2. SCAN DIV2K 2K/4K AUTHENTIC HIGH-RES IMAGES
div2k_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation/div2k_extracted")
if div2k_dir.exists():
    print("Scanning DIV2K 2K/4K authentic high-res photography...")
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for p in div2k_dir.rglob(ext):
            p_str = str(p)
            real_count += 1
            src = "DIV2K_2K4K_Authentic"
            sources[src] = sources.get(src, 0) + 1
            master_samples.append({
                "sample_id": f"DIV2K_{len(master_samples):07d}",
                "canonical_path": p_str,
                "label": 0,
                "source": src,
                "split": "TRAIN"
            })

# 3. SCAN PORTRAIT REMEDIATION POOL
portrait_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation")
if portrait_dir.exists():
    print("Scanning Portrait Remediation Pool...")
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for p in portrait_dir.rglob(ext):
            p_str = str(p)
            if "div2k_extracted" in p_str or "tmp_div2k" in p_str:
                continue
            lbl = 1 if any(k in p_str.lower() for k in ["aigc", "synthetic", "fake", "deepfake"]) else 0
            if lbl == 0:
                real_count += 1
            else:
                aigc_count += 1
            src = "Portrait_Remediation_Pool"
            sources[src] = sources.get(src, 0) + 1
            master_samples.append({
                "sample_id": f"PORT_{len(master_samples):07d}",
                "canonical_path": p_str,
                "label": lbl,
                "source": src,
                "split": "TRAIN"
            })

# 4. SCAN OLD GOVERNED BASE (20K BALANCED BASE)
old_manifest = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest_v6.jsonl"
if os.path.exists(old_manifest):
    print("Scanning Old Governed Train Base...")
    old_real, old_aigc = 0, 0
    with open(old_manifest, "r") as f:
        for line in f:
            s = json.loads(line)
            p = s.get("canonical_path", "")
            lbl = int(s.get("label", 0))
            if os.path.exists(p):
                if lbl == 0 and old_real < 10000:
                    master_samples.append({"sample_id": f"OLD_{len(master_samples):07d}", "canonical_path": p, "label": 0, "source": "Old_Governed_Real", "split": "TRAIN"})
                    old_real += 1
                    real_count += 1
                elif lbl == 1 and old_aigc < 10000:
                    master_samples.append({"sample_id": f"OLD_{len(master_samples):07d}", "canonical_path": p, "label": 1, "source": "Old_Governed_AIGC", "split": "TRAIN"})
                    old_aigc += 1
                    aigc_count += 1
            if old_real >= 10000 and old_aigc >= 10000:
                break
    sources["Old_Governed_Train_Pool"] = old_real + old_aigc

print(f"\n=====================================================================================")
print(f"Total Master Dataset Pool: {len(master_samples)} images")
print(f"  REAL Images : {real_count} ({real_count/max(len(master_samples),1)*100:.1f}%)")
print(f"  AIGC Images : {aigc_count} ({aigc_count/max(len(master_samples),1)*100:.1f}%)")
print(f"  Sources: {sources}")
print("=====================================================================================")

os.makedirs("/home/manan/aigc_robust_detection/reports", exist_ok=True)
manifest_out = "/home/manan/aigc_robust_detection/reports/master_highres_train_manifest.json"
with open(manifest_out, "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_samples": len(master_samples),
        "real_count": real_count,
        "aigc_count": aigc_count,
        "sources": sources,
        "samples": master_samples
    }, f, indent=2)

print(f"Saved Master Training Manifest to: {manifest_out}")
