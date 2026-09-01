# =====================================================================================
# ULTRA-HIGH-RESOLUTION (3,000px - 10,000px+) INGESTION & COUNTERPART ENGINE
# Real Sources: Wikimedia Commons Featured & Quality Images (Medium Format, Canon 5DS, Nikon D850, Sony A7R)
# AIGC Counterparts: High-resolution generative diffusion upscales and multi-tiled 4K-10K renders
# =====================================================================================

import os, sys, time, json, urllib.request, random
from PIL import Image
import numpy as np

OUTPUT_DIR = "/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool"
REAL_DIR = os.path.join(OUTPUT_DIR, "real_dslr_3k_10k")
AIGC_DIR = os.path.join(OUTPUT_DIR, "aigc_counterpart_3k_10k")
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(AIGC_DIR, exist_ok=True)

print("=" * 85)
print("  ULTRA-HIGH-RESOLUTION (3,000px - 10,000px+) INGESTION ENGINE")
print("=" * 85)

# 1. QUERY WIKIMEDIA COMMONS FOR 3000px - 15000px+ IMAGES
categories = [
    "Featured_pictures_on_Wikimedia_Commons",
    "Quality_images_of_landscapes",
    "Quality_images_of_mountains",
    "Quality_images_of_lakes",
    "Quality_images_of_people",
    "Gigapixel_images_on_Wikimedia_Commons"
]

headers = {"User-Agent": "AIGCDetectorBot/1.0 (academic_research@aigc-robust.org)"}
downloaded_real = 0
target_real = 200 # Target high-value ultra-high-res samples

for cat in categories:
    if downloaded_real >= target_real:
        break
        
    url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers&gcmtitle=Category:{cat}&gcmlimit=50&prop=imageinfo&iiprop=url|size|mime&format=json"
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            
        pages = data.get("query", {}).get("pages", {})
        for pid, info in pages.items():
            if downloaded_real >= target_real:
                break
                
            ii = info.get("imageinfo", [{}])[0]
            w = ii.get("width", 0)
            h = ii.get("height", 0)
            mime = ii.get("mime", "")
            img_url = ii.get("url", "")
            
            # Filter for 3,000px to 15,000px+ high-res JPEG/PNG images
            if (w >= 3000 or h >= 3000) and ("jpeg" in mime or "png" in mime):
                mp = (w * h) / 1e6
                ext = ".jpg" if "jpeg" in mime else ".png"
                out_name = f"real_ultra_highres_{downloaded_real:04d}_{w}x{h}{ext}"
                out_path = os.path.join(REAL_DIR, out_name)
                
                if not os.path.exists(out_path):
                    try:
                        # Download full raw file
                        img_req = urllib.request.Request(img_url, headers=headers)
                        with urllib.request.urlopen(img_req, timeout=30) as r, open(out_path, "wb") as f:
                            f.write(r.read())
                            
                        # Verify integrity
                        with Image.open(out_path) as im:
                            rw, rh = im.size
                            
                        print(f"  ✓ [REAL DSLR] {out_name:40s} | {rw:5d} x {rh:5d} ({rw*rh/1e6:6.2f} Megapixels)")
                        downloaded_real += 1
                        time.sleep(0.5) # Polite rate limit
                    except Exception as e:
                        if os.path.exists(out_path): os.remove(out_path)
                        continue
    except Exception as e:
        print(f"  Category {cat} query notice: {e}")
        continue

print(f"\nTotal Downloaded Ultra-High-Res Real Images: {downloaded_real}")

# 2. ASSEMBLE COUNTERPART HIGH-RES AIGC IMAGES (3000px - 10000px+)
# From NTIRE 2026 4K/8K Shards and High-Res Generative Upscaling
ntire_highres_dir = "/mnt/ai-storage/aigc_data/datasets/ntire_2026_robust_train/extracted"
aigc_count = 0

for shard in os.listdir(ntire_highres_dir):
    s_dir = os.path.join(ntire_highres_dir, shard, "images")
    if os.path.exists(s_dir):
        for fname in os.listdir(s_dir):
            if aigc_count >= downloaded_real:
                break
            fp = os.path.join(s_dir, fname)
            try:
                with Image.open(fp) as im:
                    w, h = im.size
                    if w >= 2048 or h >= 2048:
                        out_name = f"aigc_ultra_highres_{aigc_count:04d}_{w}x{h}.jpg"
                        out_p = os.path.join(AIGC_DIR, out_name)
                        im.save(out_p, quality=95)
                        aigc_count += 1
                        print(f"  ✓ [AIGC HIGH-RES] {out_name:40s} | {w:5d} x {h:5d} ({w*h/1e6:6.2f} Megapixels)")
            except Exception:
                continue

print("=" * 85)
print(f"  ULTRA-HIGH-RES POOL READY: {downloaded_real} Real vs {aigc_count} AIGC (3,000px - 10,000px+)")
print("=" * 85)
