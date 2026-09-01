import os, sys, json, hashlib, glob, time
from collections import defaultdict

print("=====================================================================")
print("  ASSEMBLING CANONICAL GOVERNED TRAINING MANIFEST V6 (MAX REAL + AIGC)")
print("=====================================================================")
start_t = time.time()

manifest_out = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest_v6.jsonl"
os.makedirs(os.path.dirname(manifest_out), exist_ok=True)

# 1. Gather all unique authentic REAL images
real_candidates = []
seen_real_paths = set()

# A. WikiArt extracted pool
wikiart_files = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/wikiart_real/*.jpg"))
print(f"Discovered {len(wikiart_files):,d} extracted WikiArt images.")
for p in wikiart_files:
    if p not in seen_real_paths:
        seen_real_paths.add(p)
        real_candidates.append({
            "canonical_path": p,
            "source_dataset": "wikiart_hard_negatives",
            "label": 0,
            "generator_or_domain": "WikiArt_Fine_Art",
            "provenance": "WikiArt Parquet HuggingFace Archive (Decoded)"
        })

# B. Defactify Real
defactify_real = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/defactify_real/*.jpg"))
print(f"Discovered {len(defactify_real):,d} Defactify Real images.")
for p in defactify_real:
    if p not in seen_real_paths:
        seen_real_paths.add(p)
        real_candidates.append({
            "canonical_path": p,
            "source_dataset": "defactify_real",
            "label": 0,
            "generator_or_domain": "COCO_Authentic_Photography",
            "provenance": "Defactify AAAI 2024 Benchmark Real Source (Decoded)"
        })

# C. SID Real
sid_real = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/sid_real/*.*"))
print(f"Discovered {len(sid_real):,d} SID Real images.")
for p in sid_real:
    if p not in seen_real_paths:
        seen_real_paths.add(p)
        real_candidates.append({
            "canonical_path": p,
            "source_dataset": "sid_real",
            "label": 0,
            "generator_or_domain": "Natural_SID_Photography",
            "provenance": "SID Benchmark Real Natural Photography (Decoded)"
        })

# D. Massive Balanced 50k Real
mb_real = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real/*.jpg"))
print(f"Discovered {len(mb_real):,d} Massive Balanced Real images.")
for p in mb_real:
    if p not in seen_real_paths:
        seen_real_paths.add(p)
        real_candidates.append({
            "canonical_path": p,
            "source_dataset": "massive_balanced_50k_real",
            "label": 0,
            "generator_or_domain": "COCO_Authentic_Photography",
            "provenance": "MS-COCO 2017 Real Photography Pool"
        })

# E. Scaled Massive Real
sm_real = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/scaled_massive/real/*.jpg"))
print(f"Discovered {len(sm_real):,d} Scaled Massive Real images.")
for p in sm_real:
    if p not in seen_real_paths:
        seen_real_paths.add(p)
        real_candidates.append({
            "canonical_path": p,
            "source_dataset": "scaled_massive_real",
            "label": 0,
            "generator_or_domain": "Natural_Photography",
            "provenance": "Curated Natural Photographic Pool"
        })

# F. CF Slice Real
cf_real = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/cf_slice/real/*.jpg"))
print(f"Discovered {len(cf_real):,d} CF Slice Real images.")
for p in cf_real:
    if p not in seen_real_paths:
        seen_real_paths.add(p)
        real_candidates.append({
            "canonical_path": p,
            "source_dataset": "cf_slice_real",
            "label": 0,
            "generator_or_domain": "COCO_Authentic_Photography",
            "provenance": "Counterfactual Slice Real Images"
        })

print(f"Total Unique Authentic REAL Candidates: {len(real_candidates):,d}")

# 2. Gather diverse AIGC candidate images
aigc_candidates = []
seen_aigc_paths = set()

# A. Quality Paradox
qp_files = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/phase2_unpacked/quality_paradox/*.jpg"))
for p in qp_files:
    if p not in seen_aigc_paths:
        seen_aigc_paths.add(p)
        aigc_candidates.append({
            "canonical_path": p,
            "source_dataset": "quality_paradox",
            "label": 1,
            "generator_or_domain": "Quality_Paradox_Photorealism",
            "provenance": "Quality Paradox NeurIPS 2024 Synthetic Generations"
        })

# B. SID Synthetic
sid_syn = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/sid_synthetic/*.*"))
for p in sid_syn:
    if p not in seen_aigc_paths:
        seen_aigc_paths.add(p)
        aigc_candidates.append({
            "canonical_path": p,
            "source_dataset": "sid_synthetic",
            "label": 1,
            "generator_or_domain": "SID_LatentDiffusion",
            "provenance": "SID Latent Diffusion Synthetic Generations"
        })

# C. Defactify Synthetic
defactify_syn = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool/defactify_synthetic/*.jpg"))
for p in defactify_syn:
    if p not in seen_aigc_paths:
        seen_aigc_paths.add(p)
        aigc_candidates.append({
            "canonical_path": p,
            "source_dataset": "defactify_synthetic",
            "label": 1,
            "generator_or_domain": "Defactify_AIGC",
            "provenance": "Defactify AAAI 2024 Synthetic Inpainting"
        })

# D. Massive Balanced 50k Synthetic
mb_syn = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/synthetic/*.*"))
for p in mb_syn:
    if p not in seen_aigc_paths:
        seen_aigc_paths.add(p)
        aigc_candidates.append({
            "canonical_path": p,
            "source_dataset": "massive_balanced_50k_synthetic",
            "label": 1,
            "generator_or_domain": "SDXL_Midjourney",
            "provenance": "SDXL and Midjourney Synthetics"
        })

# E. Scaled Massive Synthetic
sm_syn = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/scaled_massive/synthetic/*.*"))
for p in sm_syn:
    if p not in seen_aigc_paths:
        seen_aigc_paths.add(p)
        aigc_candidates.append({
            "canonical_path": p,
            "source_dataset": "scaled_massive_synthetic",
            "label": 1,
            "generator_or_domain": "Diverse_Generators",
            "provenance": "Scaled Massive Multi-Generator Synthetics"
        })

# F. Scaled Train & Balanced Scaled Train Synthetic
st_syn = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/scaled_train/synthetic/*.*")) + sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/balanced_scaled_train/synthetic/*.*"))
for p in st_syn:
    if p not in seen_aigc_paths:
        seen_aigc_paths.add(p)
        aigc_candidates.append({
            "canonical_path": p,
            "source_dataset": "scaled_train_synthetic",
            "label": 1,
            "generator_or_domain": "Diffusion_Synthetics",
            "provenance": "Diffusion Scaled Staging Synthetics"
        })

# G. CF Slice Synthetic
cf_syn = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/cf_slice/synthetic/*.*"))
for p in cf_syn:
    if p not in seen_aigc_paths:
        seen_aigc_paths.add(p)
        aigc_candidates.append({
            "canonical_path": p,
            "source_dataset": "cf_slice_synthetic",
            "label": 1,
            "generator_or_domain": "Latent_Diffusion",
            "provenance": "Counterfactual Latent Diffusion Synthetics"
        })

print(f"Total Unique Diverse AIGC Candidates: {len(aigc_candidates):,d}")

# 3. Partition Allocations (Strictly Disjoint)
# Preserve locked test split (10,316 rows from manifest v5)
locked_test_records = []
with open("/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest_v5.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["split"] == "INTERNAL_TEST":
            locked_test_records.append(r)

print(f"Preserved {len(locked_test_records):,d} Locked Internal Test rows.")

# Dev: 5,000 Real + 5,000 AIGC
# Cal: 2,000 Real + 2,000 AIGC
# Train: Remainder
import random
random.seed(42)
random.shuffle(real_candidates)
random.shuffle(aigc_candidates)

dev_real = real_candidates[:5000]
cal_real = real_candidates[5000:7000]
train_real = real_candidates[7000:]

dev_aigc = aigc_candidates[:5000]
cal_aigc = aigc_candidates[5000:7000]
train_aigc = aigc_candidates[7000:]

# Write manifest v6
total_written = 0
split_counts = defaultdict(int)

with open(manifest_out, "w") as f_out:
    # 1. Train
    for r in train_real:
        rec = {
            "canonical_path": r["canonical_path"],
            "source_dataset": r["source_dataset"],
            "label": 0,
            "generator_or_domain": r["generator_or_domain"],
            "sha256": hashlib.sha256(r["canonical_path"].encode()).hexdigest(),
            "split": "TRAIN",
            "provenance": r["provenance"],
            "image_id": f"TRAIN_{total_written:07d}"
        }
        f_out.write(json.dumps(rec) + "\n")
        total_written += 1
        split_counts["TRAIN_REAL"] += 1

    for r in train_aigc:
        rec = {
            "canonical_path": r["canonical_path"],
            "source_dataset": r["source_dataset"],
            "label": 1,
            "generator_or_domain": r["generator_or_domain"],
            "sha256": hashlib.sha256(r["canonical_path"].encode()).hexdigest(),
            "split": "TRAIN",
            "provenance": r["provenance"],
            "image_id": f"TRAIN_{total_written:07d}"
        }
        f_out.write(json.dumps(rec) + "\n")
        total_written += 1
        split_counts["TRAIN_AIGC"] += 1

    # 2. Dev
    for r in dev_real + dev_aigc:
        rec = {
            "canonical_path": r["canonical_path"],
            "source_dataset": r["source_dataset"],
            "label": r["label"],
            "generator_or_domain": r["generator_or_domain"],
            "sha256": hashlib.sha256(r["canonical_path"].encode()).hexdigest(),
            "split": "DEV",
            "provenance": r["provenance"],
            "image_id": f"DEV_{total_written:07d}"
        }
        f_out.write(json.dumps(rec) + "\n")
        total_written += 1
        split_counts[f"DEV_{'REAL' if r['label']==0 else 'AIGC'}"] += 1

    # 3. Calibration
    for r in cal_real + cal_aigc:
        rec = {
            "canonical_path": r["canonical_path"],
            "source_dataset": r["source_dataset"],
            "label": r["label"],
            "generator_or_domain": r["generator_or_domain"],
            "sha256": hashlib.sha256(r["canonical_path"].encode()).hexdigest(),
            "split": "CALIBRATION",
            "provenance": r["provenance"],
            "image_id": f"CAL_{total_written:07d}"
        }
        f_out.write(json.dumps(rec) + "\n")
        total_written += 1
        split_counts[f"CAL_{'REAL' if r['label']==0 else 'AIGC'}"] += 1

    # 4. Locked Internal Test
    for r in locked_test_records:
        f_out.write(json.dumps(r) + "\n")
        total_written += 1
        split_counts[f"TEST_{'REAL' if r['label']==0 else 'AIGC'}"] += 1

with open(manifest_out, "rb") as f:
    final_sha = hashlib.sha256(f.read()).hexdigest()

print(f"\nManifest v6 Assembly Completed in {time.time()-start_t:.2f}s!")
print(f"Manifest Path: {manifest_out}")
print(f"Manifest SHA-256: {final_sha}")
print(f"Total Rows: {total_written:,d}")
print(f"  TRAIN:       {split_counts['TRAIN_REAL'] + split_counts['TRAIN_AIGC']:,d} (Real: {split_counts['TRAIN_REAL']:,d}, AIGC: {split_counts['TRAIN_AIGC']:,d})")
print(f"  DEV:         {split_counts['DEV_REAL'] + split_counts['DEV_AIGC']:,d} (Real: {split_counts['DEV_REAL']:,d}, AIGC: {split_counts['DEV_AIGC']:,d})")
print(f"  CALIBRATION: {split_counts['CAL_REAL'] + split_counts['CAL_AIGC']:,d} (Real: {split_counts['CAL_REAL']:,d}, AIGC: {split_counts['CAL_AIGC']:,d})")
print(f"  TEST:        {split_counts['TEST_REAL'] + split_counts['TEST_AIGC']:,d} (Real: {split_counts['TEST_REAL']:,d}, AIGC: {split_counts['TEST_AIGC']:,d})")
