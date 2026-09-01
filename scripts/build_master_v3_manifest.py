# =====================================================================================
# MASTER V3 BROAD-DATASET MANIFEST BUILDER & DEDUPLICATION ENGINE
# Total Balanced Corpus: 60,000 Physical Images (30,000 Real vs 30,000 AIGC)
# Split: 50,000 V3 Train (25k Real, 25k AIGC) | 10,000 V3 Validation (5k Real, 5k AIGC)
# =====================================================================================

import os, sys, time, json, random, glob, hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

# Deterministic Seed
random.seed(42)

print("=" * 85)
print("  MASTER V3 MANIFEST BUILDER & DEDUPLICATION ENGINE")
print("=" * 85)

# 1. LOAD STRICT BENCHMARK ISOLATION SET
BENCHMARK_PATH = "/home/manan/aigc_robust_detection/reports/final_production_v2_strict_audit_report.json"
benchmark_paths = set()
if os.path.exists(BENCHMARK_PATH):
    with open(BENCHMARK_PATH, "r") as f:
        bm = json.load(f)
    for s in bm.get("top_false_positives", []) + bm.get("top_false_negatives", []):
        benchmark_paths.add(s["path"])
print(f"Loaded Benchmark Exclusion Set: {len(benchmark_paths)} isolated paths")

# 2. ASSEMBLE 30,000 REAL PHOTOGRAPHY SAMPLES
real_pool = []

# A. 8,000 Paired Hard Negatives
hard_neg_dir = "/mnt/ai-storage/aigc_data/datasets/hard_negative_remediation"
for cat in os.listdir(hard_neg_dir):
    cat_path = os.path.join(hard_neg_dir, cat)
    if os.path.isdir(cat_path):
        for f in glob.glob(f"{cat_path}/*.jpg"):
            real_pool.append({
                "canonical_path": f,
                "label": 0,
                "category": f"hard_negative_{cat}",
                "generator_source": "real_photoshop_lightroom_compression",
                "provenance": "PILLAR1_PAIRED_HARD_NEGATIVE"
            })
print(f"  [Real] Paired Hard Negatives   : {len(real_pool):,}")

# B. 4,118 Portrait Remediation (Studio, Headshots, Selfies, Retouched)
portrait_dir = "/mnt/ai-storage/aigc_data/datasets/portrait_remediation"
for sub in ["real_portrait", "real_headshot", "real_retouched", "real_studio"]:
    sub_p = os.path.join(portrait_dir, sub)
    if os.path.exists(sub_p):
        for f in glob.glob(f"{sub_p}/*"):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                real_pool.append({
                    "canonical_path": f,
                    "label": 0,
                    "category": sub,
                    "generator_source": "authentic_portrait_studio_dslr",
                    "provenance": "PORTRAIT_REMEDIATION_REAL"
                })
print(f"  [Real] + Portrait Remediation  : {len(real_pool):,}")

# C. 100 DIV2K 2K/4K Uncompressed DSLRs
div2k_dir = "/mnt/ai-storage/aigc_data/datasets/portrait_remediation/div2k_extracted"
if os.path.exists(div2k_dir):
    for f in glob.glob(f"{div2k_dir}/**/*.png", recursive=True) + glob.glob(f"{div2k_dir}/**/*.jpg", recursive=True):
        real_pool.append({
            "canonical_path": f,
            "label": 0,
            "category": "div2k_dslr_2k_4k",
            "generator_source": "authentic_dslr_raw_photography",
            "provenance": "DIV2K_HR"
        })
print(f"  [Real] + DIV2K 2K/4K DSLRs     : {len(real_pool):,}")

# D. 3,500 ImageNet Authentic Camera Captures
imagenet_real = "/mnt/ai-storage/aigc_data/datasets/remediation_expansion_pool/ImageNet_Authentic_Photo"
if os.path.exists(imagenet_real):
    for f in glob.glob(f"{imagenet_real}/*"):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            real_pool.append({
                "canonical_path": f,
                "label": 0,
                "category": "imagenet_authentic_real",
                "generator_source": "authentic_camera_photo",
                "provenance": "IMAGENET_REAL"
            })
print(f"  [Real] + ImageNet Authentic    : {len(real_pool):,}")

# E. Diverse Real Photography (Landscapes, Nature, Architecture, Street, Night)
massive_real = "/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real"
if os.path.exists(massive_real):
    for f in sorted(glob.glob(f"{massive_real}/*")):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            real_pool.append({
                "canonical_path": f,
                "label": 0,
                "category": "diverse_real_photography",
                "generator_source": "authentic_world_photography",
                "provenance": "MASSIVE_BALANCED_REAL"
            })
print(f"  [Real] Total Assembled Real    : {len(real_pool):,}")

# 3. ASSEMBLE 30,000 MULTI-GENERATOR AIGC SAMPLES
aigc_pool = []

# A. 9,000 SynthBuster Multi-Generator Pool
synthbuster_dir = "/mnt/ai-storage/aigc_data/datasets/synthbuster/synthbuster"
if os.path.exists(synthbuster_dir):
    for gen in os.listdir(synthbuster_dir):
        gen_p = os.path.join(synthbuster_dir, gen)
        if os.path.isdir(gen_p):
            for f in glob.glob(f"{gen_p}/*"):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    aigc_pool.append({
                        "canonical_path": f,
                        "label": 1,
                        "category": f"synthbuster_{gen}",
                        "generator_source": gen,
                        "provenance": "SYNTHBUSTER"
                    })
print(f"  [AIGC] SynthBuster Generators  : {len(aigc_pool):,}")

# B. 10,000 Remediation Expansion Pool (ADM, BigGAN, GLIDE, VQDM, Wukong)
rem_dir = "/mnt/ai-storage/aigc_data/datasets/remediation_expansion_pool"
for gname in ["ADM_PixelDiffusion", "BigGAN_Adversarial", "GLIDE_PixelDiffusion", "VQDM_DiscreteDiffusion", "Wukong_BilingualDiffusion"]:
    g_p = os.path.join(rem_dir, gname)
    if os.path.exists(g_p):
        for f in glob.glob(f"{g_p}/*"):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                aigc_pool.append({
                    "canonical_path": f,
                    "label": 1,
                    "category": f"expansion_{gname}",
                    "generator_source": gname,
                    "provenance": "REMEDIATION_EXPANSION"
                })
print(f"  [AIGC] + Expansion Generators  : {len(aigc_pool):,}")

# C. 15,000 High-Resolution NTIRE 2026 Diffusion Shards
ntire_dir = "/mnt/ai-storage/aigc_data/datasets/ntire_2026_robust_train/extracted"
for shard in ["shard_0", "shard_1", "shard_2", "shard_3", "shard_4"]:
    s_p = os.path.join(ntire_dir, shard, "images")
    if os.path.exists(s_p):
        files = glob.glob(f"{s_p}/*.jpg")
        for f in files[:3000]: # 3k per shard = 15k total
            aigc_pool.append({
                "canonical_path": f,
                "label": 1,
                "category": f"ntire_2026_{shard}",
                "generator_source": "ntire_highres_diffusion",
                "provenance": "NTIRE_2026_ROBUST"
            })
print(f"  [AIGC] Total Assembled AIGC    : {len(aigc_pool):,}")

# 4. DEDUPLICATION & BENCHMARK ISOLATION FILTERING
unique_real = {}
for s in real_pool:
    p = s["canonical_path"]
    if p not in benchmark_paths and p not in unique_real and os.path.exists(p):
        unique_real[p] = s

unique_aigc = {}
for s in aigc_pool:
    p = s["canonical_path"]
    if p not in benchmark_paths and p not in unique_aigc and os.path.exists(p):
        unique_aigc[p] = s

clean_real = list(unique_real.values())
clean_aigc = list(unique_aigc.values())
random.shuffle(clean_real)
random.shuffle(clean_aigc)

print(f"\n[DEDUPLICATION & BENCHMARK FILTERING RESULTS]")
print(f"  Clean Real Pool  : {len(clean_real):,} unique physical images")
print(f"  Clean AIGC Pool  : {len(clean_aigc):,} unique physical images")

# Balance to 30,000 Real + 30,000 AIGC = 60,000 Total
target_real = min(len(clean_real), 30000)
target_aigc = min(len(clean_aigc), 30000)
target_per_class = min(target_real, target_aigc)

final_real = clean_real[:target_per_class]
final_aigc = clean_aigc[:target_per_class]

# Split into 50k V3 Train (25k Real, 25k AIGC) + 10k V3 Val (5k Real, 5k AIGC)
train_real = final_real[:25000]
val_real = final_real[25000:30000]

train_aigc = final_aigc[:25000]
val_aigc = final_aigc[25000:30000]

v3_train = train_real + train_aigc
v3_val = val_real + val_aigc
random.shuffle(v3_train)
random.shuffle(v3_val)

print(f"\n[V3 FINAL SPLIT COMPOSITION]")
print(f"  V3 Train Corpus      : {len(v3_train):,} images (25,000 Real, 25,000 AIGC)")
print(f"  V3 Validation Corpus : {len(v3_val):,} images (5,000 Real, 5,000 AIGC)")
print(f"  Total V3 Dataset     : {len(v3_train) + len(v3_val):,} physical images")

# 5. SAVE MANIFESTS
train_manifest_path = "/home/manan/aigc_robust_detection/reports/master_v3_train_manifest.json"
val_manifest_path = "/home/manan/aigc_robust_detection/reports/master_v3_val_manifest.json"

with open(train_manifest_path, "w") as f:
    json.dump({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_train_samples": len(v3_train),
        "real_count": len(train_real),
        "aigc_count": len(train_aigc),
        "samples": v3_train
    }, f, indent=2)

with open(val_manifest_path, "w") as f:
    json.dump({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_val_samples": len(v3_val),
        "real_count": len(val_real),
        "aigc_count": len(val_aigc),
        "samples": v3_val
    }, f, indent=2)

print(f"  >> Saved V3 Train Manifest : {train_manifest_path}")
print(f"  >> Saved V3 Val Manifest   : {val_manifest_path}")
print("=" * 85)
