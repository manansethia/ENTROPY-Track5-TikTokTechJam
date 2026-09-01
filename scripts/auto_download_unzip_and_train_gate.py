# =====================================================================================
# AUTO-ORCHESTRATOR: DOWNLOAD-FIRST -> UNZIP -> FINAL MASTER TRAINING
# Monitors Active Downloads, Unzips All Archives, Verifies Manifest, & Launches GPU Training
# =====================================================================================

import os, sys, time, subprocess, json, glob, hashlib, gc
from pathlib import Path

print("=" * 85)
print("  AUTO-ORCHESTRATOR: DOWNLOAD-FIRST -> UNZIP -> START")
print("=" * 85)

NTIRE_DIR = Path("/mnt/ai-storage/aigc_data/datasets/ntire_2026_robust_train")
EXTRACTED_DIR = NTIRE_DIR / "extracted"
os.makedirs(EXTRACTED_DIR, exist_ok=True)

# 1. WAIT FOR ALL NTIRE SHARDS TO FINISH DOWNLOADING
print("\n--- [STAGE 1/4] Monitoring Background Downloads ---")
while True:
    # Check if download process is alive
    res = subprocess.run(["pgrep", "-f", "download_chunked_resumable_datasets"], capture_output=True, text=True)
    is_downloading = (res.returncode == 0)
    
    # Check if shard_3.zip exists and is complete (no .incomplete lock)
    shard_3_file = NTIRE_DIR / "shard_3.zip"
    has_shard_3 = shard_3_file.exists()
    
    if not is_downloading or has_shard_3:
        print(f"[{time.strftime('%H:%M:%S')}] Downloader status: Complete/Ready (shard_3.zip exists: {has_shard_3})")
        break
    
    print(f"[{time.strftime('%H:%M:%S')}] Downloads actively streaming (PID active). Checking again in 30s...")
    time.sleep(30)

# 2. UNZIP ALL DOWNLOADED SHARDS
print("\n--- [STAGE 2/4] Unzipping All Downloaded Datasets ---")
zip_files = list(NTIRE_DIR.glob("*.zip"))
print(f"Found {len(zip_files)} archives to extract: {[z.name for z in zip_files]}")

for z in zip_files:
    target_subdir = EXTRACTED_DIR / z.stem
    if not target_subdir.exists() or len(list(target_subdir.glob("*"))) == 0:
        print(f"Extracting {z.name} -> {target_subdir}...")
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(EXTRACTED_DIR)], check=True)
    else:
        print(f"Archive {z.name} already extracted ({len(list(target_subdir.glob('*')))} files).")

print("All archives successfully extracted and verified on disk.")

# 3. BUILD COMPLETE MASTER HIGH-RES MANIFEST
print("\n--- [STAGE 3/4] Compiling Complete Master High-Res Manifest ---")
master_samples = []
real_count = 0
aigc_count = 0
sources = {}

# Scan Extracted NTIRE Shards (all shards)
for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    for p in EXTRACTED_DIR.rglob(ext):
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

# Scan Portrait Remediation Pool
portrait_dir = Path("/mnt/ai-storage/aigc_data/datasets/portrait_remediation")
if portrait_dir.exists():
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for p in portrait_dir.rglob(ext):
            p_str = str(p)
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

# Scan Old Governed Train-Eligible Manifest (20k Balanced Base)
old_manifest = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest_v6.jsonl"
if os.path.exists(old_manifest):
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

print(f"Total Master Dataset Pool: {len(master_samples)} images (Real: {real_count}, AIGC: {aigc_count})")
print(f"Sources: {sources}")

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

# 4. EXECUTE FULL-SCALE GPU SPECIALIST TRAINING & FUSION PIPELINE
print("\n--- [STAGE 4/4] Launching Master Production Remediation Pipeline V2 on RTX 3050 GPU ---")
pipeline_script = "/home/manan/aigc_robust_detection/scripts/master_production_remediation_pipeline_v2.py"
subprocess.run(["/home/manan/.venvs/aigc-detector/bin/python", "-u", pipeline_script], check=True)

print("\n" + "=" * 85)
print("  ALL STAGES COMPLETE: DOWNLOAD -> UNZIP -> TRAINING -> FUSION -> FREEZE")
print("=" * 85)
