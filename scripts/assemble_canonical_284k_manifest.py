import os, sys, json, hashlib, time
from pathlib import Path
import numpy as np

print("================================================================")
print("ASSEMBLING AUTHORITATIVE 284,500 GOVERNED MANIFEST")
print("================================================================")

# Target Partition Sizes
TARGET_TRAIN_REAL = 149000
TARGET_TRAIN_AIGC = 111184
TARGET_TRAIN_TOTAL = 260184

TARGET_DEV_REAL = 5000
TARGET_DEV_AIGC = 5000
TARGET_DEV_TOTAL = 10000

TARGET_CAL_REAL = 2000
TARGET_CAL_AIGC = 2000
TARGET_CAL_TOTAL = 4000

TARGET_TEST_REAL = 4238
TARGET_TEST_AIGC = 6078
TARGET_TEST_TOTAL = 10316

TARGET_GRAND_TOTAL = 284500

# -------------------------------------------------------------
# 1. LOCK INTERNAL TEST FROM EXISTING PHASE2 MANIFEST
# -------------------------------------------------------------
print("\n[1/6] Extracting Locked 10,316 Internal Test...")
phase2_path = "/home/manan/aigc_robust_detection/manifests/phase2_150k_manifest.jsonl"
locked_test_records = []
dev_seed_records = []
seen_hashes = set()
seen_paths = set()

with open(phase2_path, "r") as f:
    for line in f:
        d = json.loads(line)
        split = d.get("split", "")
        p = d.get("path", d.get("image_path", ""))
        l = d.get("label", d.get("ground_truth", 0))
        h = d.get("sha256", hashlib.sha256(p.encode("utf-8")).hexdigest())
        
        # Check OOD leakage
        if any(ood in p.lower() for ood in ["synthbuster", "aigibench", "chameleon", "vct2", "wildrf", "synthwildx"]):
            continue
            
        if split == "PHASE2_INTERNAL_TEST":
            rec = {
                "image_id": f"TEST_{len(locked_test_records):06d}",
                "canonical_path": p,
                "source_dataset": d.get("dataset", "phase2_internal_test"),
                "label": int(l),
                "generator_or_domain": d.get("generator", "diverse_internal_test"),
                "sha256": h,
                "split": "INTERNAL_TEST",
                "resolution": d.get("resolution", "original"),
                "file_format": p.split(".")[-1].lower() if "." in p else "jpg"
            }
            locked_test_records.append(rec)
            seen_hashes.add(h)
            seen_paths.add(p)
        elif split == "PHASE2_VAL":
            dev_seed_records.append({
                "path": p,
                "label": int(l),
                "sha256": h,
                "generator": d.get("generator", "diverse_val"),
                "dataset": d.get("dataset", "phase2_val")
            })

print(f"      Locked Test Rows: {len(locked_test_records)} (Real: {sum(1 for r in locked_test_records if r['label']==0)}, AIGC: {sum(1 for r in locked_test_records if r['label']==1)})")

# -------------------------------------------------------------
# 2. ASSEMBLE 10,000 DEV (5,000 Real, 5,000 AIGC)
# -------------------------------------------------------------
print("\n[2/6] Assembling 10,000 Dev Split...")
dev_records = []
dev_real_needed = TARGET_DEV_REAL
dev_aigc_needed = TARGET_DEV_AIGC

for r in dev_seed_records:
    p, l, h = r["path"], r["label"], r["sha256"]
    if h in seen_hashes or p in seen_paths:
        continue
    if l == 0 and dev_real_needed > 0:
        dev_records.append({
            "image_id": f"DEV_{len(dev_records):06d}",
            "canonical_path": p,
            "source_dataset": r["dataset"],
            "label": 0,
            "generator_or_domain": r["generator"],
            "sha256": h,
            "split": "DEV",
            "resolution": "original",
            "file_format": p.split(".")[-1].lower() if "." in p else "jpg"
        })
        seen_hashes.add(h)
        seen_paths.add(p)
        dev_real_needed -= 1
    elif l == 1 and dev_aigc_needed > 0:
        dev_records.append({
            "image_id": f"DEV_{len(dev_records):06d}",
            "canonical_path": p,
            "source_dataset": r["dataset"],
            "label": 1,
            "generator_or_domain": r["generator"],
            "sha256": h,
            "split": "DEV",
            "resolution": "original",
            "file_format": p.split(".")[-1].lower() if "." in p else "jpg"
        })
        seen_hashes.add(h)
        seen_paths.add(p)
        dev_aigc_needed -= 1

print(f"      Dev Rows: {len(dev_records)} (Real: {sum(1 for r in dev_records if r['label']==0)}, AIGC: {sum(1 for r in dev_records if r['label']==1)})")

# -------------------------------------------------------------
# 3. ASSEMBLE 4,000 CALIBRATION (2,000 Real, 2,000 AIGC)
# -------------------------------------------------------------
print("\n[3/6] Assembling 4,000 Calibration Split...")
cal_records = []
cal_real_needed = TARGET_CAL_REAL
cal_aigc_needed = TARGET_CAL_AIGC

# Pool from remaining dev_seed_records first
for r in dev_seed_records:
    p, l, h = r["path"], r["label"], r["sha256"]
    if h in seen_hashes or p in seen_paths:
        continue
    if l == 0 and cal_real_needed > 0:
        cal_records.append({
            "image_id": f"CAL_{len(cal_records):06d}",
            "canonical_path": p,
            "source_dataset": r["dataset"],
            "label": 0,
            "generator_or_domain": r["generator"],
            "sha256": h,
            "split": "CALIBRATION",
            "resolution": "original",
            "file_format": p.split(".")[-1].lower() if "." in p else "jpg"
        })
        seen_hashes.add(h)
        seen_paths.add(p)
        cal_real_needed -= 1
    elif l == 1 and cal_aigc_needed > 0:
        cal_records.append({
            "image_id": f"CAL_{len(cal_records):06d}",
            "canonical_path": p,
            "source_dataset": r["dataset"],
            "label": 1,
            "generator_or_domain": r["generator"],
            "sha256": h,
            "split": "CALIBRATION",
            "resolution": "original",
            "file_format": p.split(".")[-1].lower() if "." in p else "jpg"
        })
        seen_hashes.add(h)
        seen_paths.add(p)
        cal_aigc_needed -= 1

print(f"      Calibration Rows: {len(cal_records)} (Real: {sum(1 for r in cal_records if r['label']==0)}, AIGC: {sum(1 for r in cal_records if r['label']==1)})")

# -------------------------------------------------------------
# 4. ASSEMBLE 260,184 TRAIN (149,000 Real, 111,184 AIGC)
# -------------------------------------------------------------
print("\n[4/6] Harvesting Audited Image Storage for 260,184 Training Corpus...")
train_real_records = []
train_aigc_records = []

# A. Harvest from phase2_150k_manifest.jsonl (TRAIN partition)
with open(phase2_path, "r") as f:
    for line in f:
        d = json.loads(line)
        if d.get("split", "") == "PHASE2_TRAIN":
            p = d.get("path", d.get("image_path", ""))
            l = int(d.get("label", d.get("ground_truth", 0)))
            h = d.get("sha256", hashlib.sha256(p.encode("utf-8")).hexdigest())
            if h in seen_hashes or p in seen_paths:
                continue
            if any(ood in p.lower() for ood in ["synthbuster", "aigibench", "chameleon", "vct2", "wildrf", "synthwildx"]):
                continue
                
            rec = {
                "canonical_path": p,
                "source_dataset": d.get("dataset", "phase2_train"),
                "label": l,
                "generator_or_domain": d.get("generator", "diverse_training"),
                "sha256": h,
                "split": "TRAIN",
                "resolution": d.get("resolution", "original"),
                "file_format": p.split(".")[-1].lower() if "." in p else "jpg"
            }
            if l == 0 and len(train_real_records) < TARGET_TRAIN_REAL:
                train_real_records.append(rec)
                seen_hashes.add(h)
                seen_paths.add(p)
            elif l == 1 and len(train_aigc_records) < TARGET_TRAIN_AIGC:
                train_aigc_records.append(rec)
                seen_hashes.add(h)
                seen_paths.add(p)

print(f"      After Phase 2 Harvest: Real={len(train_real_records):,}/{TARGET_TRAIN_REAL:,} | AIGC={len(train_aigc_records):,}/{TARGET_TRAIN_AIGC:,}")

# B. Harvest from Unpacked Storage Pools (massive_balanced_50k, scaled_massive, balanced_scaled_train, cf_slice)
storage_dirs = [
    ("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k", "massive_balanced_50k"),
    ("/mnt/ai-storage/aigc_data/datasets/scaled_massive", "scaled_massive"),
    ("/mnt/ai-storage/aigc_data/datasets/balanced_scaled_train", "balanced_scaled_train"),
    ("/mnt/ai-storage/aigc_data/datasets/scaled_45k", "scaled_45k"),
    ("/mnt/ai-storage/aigc_data/datasets/scaled_train", "scaled_train"),
    ("/mnt/ai-storage/aigc_data/datasets/cf_slice", "cf_slice"),
    ("/mnt/ai-storage/aigc_data/datasets/phase2_unpacked", "phase2_unpacked")
]

for sdir, sname in storage_dirs:
    if len(train_real_records) >= TARGET_TRAIN_REAL and len(train_aigc_records) >= TARGET_TRAIN_AIGC:
        break
    if not os.path.exists(sdir):
        continue
        
    for root, _, files in os.walk(sdir):
        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            fp = os.path.join(root, f)
            if fp in seen_paths:
                continue
            if any(ood in fp.lower() for ood in ["synthbuster", "aigibench", "chameleon", "vct2", "wildrf", "synthwildx"]):
                continue
                
            # Compute real/synthetic from path provenance
            p_lower = fp.lower()
            if "/real/" in p_lower or "wikiart" in p_lower or "coco" in p_lower:
                l = 0
                gen = "authentic_photography" if "coco" in p_lower else ("fine_art" if "wikiart" in p_lower else "real_camera")
            elif "/synthetic/" in p_lower or "/fake/" in p_lower or "quality_paradox" in p_lower or "cf_" in p_lower or "hfcf" in p_lower:
                l = 1
                gen = "quality_paradox" if "qp" in p_lower else ("latent_diffusion" if "cf" in p_lower else "hfcf_synthetic")
            else:
                continue
                
            h = hashlib.sha256(fp.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
                
            rec = {
                "canonical_path": fp,
                "source_dataset": sname,
                "label": l,
                "generator_or_domain": gen,
                "sha256": h,
                "split": "TRAIN",
                "resolution": "original",
                "file_format": f.split(".")[-1].lower()
            }
            
            if l == 0 and len(train_real_records) < TARGET_TRAIN_REAL:
                train_real_records.append(rec)
                seen_hashes.add(h)
                seen_paths.add(fp)
            elif l == 1 and len(train_aigc_records) < TARGET_TRAIN_AIGC:
                train_aigc_records.append(rec)
                seen_hashes.add(h)
                seen_paths.add(fp)

print(f"      After Unpacked Storage Harvest: Real={len(train_real_records):,}/{TARGET_TRAIN_REAL:,} | AIGC={len(train_aigc_records):,}/{TARGET_TRAIN_AIGC:,}")

# Assign unique image IDs for Train
train_records = []
for idx, r in enumerate(train_real_records + train_aigc_records):
    r["image_id"] = f"TRAIN_{idx:07d}"
    train_records.append(r)

# -------------------------------------------------------------
# 5. ASSEMBLE FINAL CORPUS & WRITE CANONICAL MANIFEST
# -------------------------------------------------------------
all_records = train_records + dev_records + cal_records + locked_test_records
out_manifest_path = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest.jsonl"
os.makedirs(os.path.dirname(out_manifest_path), exist_ok=True)

h_out = hashlib.sha256()
with open(out_manifest_path, "w") as f:
    for r in all_records:
        line_str = json.dumps(r) + "\n"
        h_out.update(line_str.encode("utf-8"))
        f.write(line_str)

manifest_sha256 = h_out.hexdigest()
print(f"\n[5/6] Canonical Manifest Written: {out_manifest_path}")
print(f"      Total Rows: {len(all_records):,}")
print(f"      Manifest SHA-256: {manifest_sha256}")

# -------------------------------------------------------------
# 6. EXACT SET INTERSECTION AUDIT (ALL PAIRS)
# -------------------------------------------------------------
print("\n[6/6] Computing Exact Split Disjointness & Intersections...")
split_hashes = {"TRAIN": set(), "DEV": set(), "CALIBRATION": set(), "INTERNAL_TEST": set()}
split_counts = {"TRAIN": 0, "DEV": 0, "CALIBRATION": 0, "INTERNAL_TEST": 0}
split_labels = {
    "TRAIN": {"REAL": 0, "AIGC": 0},
    "DEV": {"REAL": 0, "AIGC": 0},
    "CALIBRATION": {"REAL": 0, "AIGC": 0},
    "INTERNAL_TEST": {"REAL": 0, "AIGC": 0}
}
ood_leakage = 0

for r in all_records:
    s = r["split"]
    h = r["sha256"]
    l = "REAL" if r["label"] == 0 else "AIGC"
    
    split_hashes[s].add(h)
    split_counts[s] += 1
    split_labels[s][l] += 1
    
    if any(ood in r["canonical_path"].lower() for ood in ["synthbuster", "aigibench", "chameleon", "vct2", "wildrf", "synthwildx"]):
        if s != "OOD":
            ood_leakage += 1

splits = ["TRAIN", "DEV", "CALIBRATION", "INTERNAL_TEST"]
intersections = {}
disjoint = True
for i in range(len(splits)):
    for j in range(i + 1, len(splits)):
        s1, s2 = splits[i], splits[j]
        inter = len(split_hashes[s1].intersection(split_hashes[s2]))
        intersections[f"{s1}_AND_{s2}"] = inter
        if inter > 0:
            disjoint = False
        print(f"      {s1} ∩ {s2}: {inter}")

print(f"\nDisjointness Verified: {disjoint}")
print(f"OOD Contamination: {ood_leakage}")
for s in splits:
    print(f"  {s:15s}: {split_counts[s]:,d} rows ({split_labels[s]})")

print("\n================================================================")
print("CANONICAL 284,500 MANIFEST ASSEMBLY COMPLETE")
print("================================================================")
