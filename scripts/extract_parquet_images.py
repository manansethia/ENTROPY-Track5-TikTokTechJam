import os, sys, glob, hashlib, io
import pyarrow.parquet as pq
from PIL import Image

out_base = "/mnt/ai-storage/aigc_data/datasets/extracted_parquet_pool"
os.makedirs(os.path.join(out_base, "defactify_real"), exist_ok=True)
os.makedirs(os.path.join(out_base, "defactify_synthetic"), exist_ok=True)
os.makedirs(os.path.join(out_base, "sid_real"), exist_ok=True)
os.makedirs(os.path.join(out_base, "sid_synthetic"), exist_ok=True)

print("=== 1. EXTRACTING DEFACTIFY REAL & SYNTHETIC IMAGES ===")
defactify_files = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/defactify/data/*.parquet"))
defact_real_count = 0
defact_synth_count = 0

for pf in defactify_files:
    t = pq.read_table(pf)
    images = t["Image"].to_pylist()
    labels = t["Label_A"].to_pylist()
    
    for idx, (img_item, label) in enumerate(zip(images, labels)):
        if isinstance(img_item, dict) and "bytes" in img_item:
            img_bytes = img_item["bytes"]
        elif isinstance(img_item, list):
            # list of tuples
            img_bytes = dict(img_item).get("bytes", None)
        else:
            continue
            
        if not img_bytes:
            continue
            
        sha = hashlib.sha256(img_bytes).hexdigest()
        if label == 0:
            target_p = os.path.join(out_base, "defactify_real", f"{sha}.jpg")
            if not os.path.exists(target_p):
                with open(target_p, "wb") as f:
                    f.write(img_bytes)
            defact_real_count += 1
        elif label == 1:
            # extract up to 5,000 for Defactify synthetic target
            if defact_synth_count < 5000:
                target_p = os.path.join(out_base, "defactify_synthetic", f"{sha}.jpg")
                if not os.path.exists(target_p):
                    with open(target_p, "wb") as f:
                        f.write(img_bytes)
                defact_synth_count += 1

print(f"Defactify Extracted: Real={defact_real_count:,} | Synthetic={defact_synth_count:,}")

print("\n=== 2. EXTRACTING SID REAL & SYNTHETIC IMAGES ===")
sid_files = sorted(glob.glob("/mnt/ai-storage/aigc_data/datasets/sid_parquet/*.parquet"))
sid_real_count = 0
sid_synth_count = 0

for pf in sid_files:
    t = pq.read_table(pf)
    images = t["image"].to_pylist()
    labels = t["label"].to_pylist()
    
    for idx, (img_item, label) in enumerate(zip(images, labels)):
        if isinstance(img_item, dict) and "bytes" in img_item:
            img_bytes = img_item["bytes"]
        elif isinstance(img_item, list):
            img_bytes = dict(img_item).get("bytes", None)
        else:
            continue
            
        if not img_bytes:
            continue
            
        sha = hashlib.sha256(img_bytes).hexdigest()
        if label == 0:
            target_p = os.path.join(out_base, "sid_real", f"{sha}.png")
            if not os.path.exists(target_p):
                with open(target_p, "wb") as f:
                    f.write(img_bytes)
            sid_real_count += 1
        elif label in (1, 2):
            if sid_synth_count < 15000:
                target_p = os.path.join(out_base, "sid_synthetic", f"{sha}.png")
                if not os.path.exists(target_p):
                    with open(target_p, "wb") as f:
                        f.write(img_bytes)
                sid_synth_count += 1

print(f"SID Extracted: Real={sid_real_count:,} | Synthetic={sid_synth_count:,}")
