# =====================================================================================
# PILLAR 1: PAIRED HARD-NEGATIVE PERTURBATION GENERATOR
# Generates realistic Photoshop, Lightroom, and Web-Compression transformations
# on authentic real photos, keeping 1:1 paired provenance tracking.
# Target: /mnt/ai-storage/aigc_data/datasets/hard_negative_remediation/
# =====================================================================================

import os, sys, time, json, random, io
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
import numpy as np

# Deterministic Seed
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "/mnt/ai-storage/aigc_data/datasets/hard_negative_remediation"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["jpeg_compressed", "webp_compressed", "downscale_upscale", "sharpened", "denoised_smoothed", "hdr_clahe", "lightroom_clarity_vibrance", "compound_social_media"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# 1. TRANSFORMATION FUNCTIONS
def apply_jpeg_compression(img: Image.Image, q: int = None) -> Image.Image:
    q = q or random.randint(40, 85)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def apply_webp_compression(img: Image.Image, q: int = None) -> Image.Image:
    q = q or random.randint(40, 85)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def apply_downscale_upscale(img: Image.Image, scale: float = None) -> Image.Image:
    scale = scale or random.uniform(0.35, 0.65)
    w, h = img.size
    dw, dh = max(32, int(w * scale)), max(32, int(h * scale))
    small = img.resize((dw, dh), resample=Image.Resampling.BILINEAR)
    return small.resize((w, h), resample=Image.Resampling.BICUBIC)

def apply_photoshop_sharpening(img: Image.Image) -> Image.Image:
    # Unsharp Mask simulation
    radius = random.uniform(1.5, 3.0)
    percent = random.randint(120, 200)
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))

def apply_denoising_smoothing(img: Image.Image) -> Image.Image:
    # Bilateral filter for skin smoothing / surface denoising
    arr = np.array(img)
    d = random.choice([7, 9, 11])
    sigma_color = random.randint(50, 100)
    sigma_space = random.randint(50, 100)
    smooth = cv2.bilateralFilter(arr, d, sigma_color, sigma_space)
    return Image.fromarray(smooth)

def apply_hdr_clahe(img: Image.Image) -> Image.Image:
    # Local contrast / HDR tone mapping
    arr = np.array(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=random.uniform(2.0, 4.0), tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

def apply_lightroom_clarity_vibrance(img: Image.Image) -> Image.Image:
    # Boost contrast and saturation
    enh_c = ImageEnhance.Contrast(img)
    img_c = enh_c.enhance(random.uniform(1.2, 1.5))
    enh_s = ImageEnhance.Color(img_c)
    img_s = enh_s.enhance(random.uniform(1.2, 1.6))
    return apply_photoshop_sharpening(img_s)

def apply_compound_social_media(img: Image.Image) -> Image.Image:
    # Compound pipeline: (Denoise/Smooth -> Tone Map -> Downscale/Upscale -> JPEG/WebP)
    step1 = apply_denoising_smoothing(img)
    step2 = apply_hdr_clahe(step1)
    step3 = apply_downscale_upscale(step2, scale=0.5)
    return apply_jpeg_compression(step3, q=random.randint(50, 75))

TRANSFORMS = {
    "jpeg_compressed": apply_jpeg_compression,
    "webp_compressed": apply_webp_compression,
    "downscale_upscale": apply_downscale_upscale,
    "sharpened": apply_photoshop_sharpening,
    "denoised_smoothed": apply_denoising_smoothing,
    "hdr_clahe": apply_hdr_clahe,
    "lightroom_clarity_vibrance": apply_lightroom_clarity_vibrance,
    "compound_social_media": apply_compound_social_media
}

# 2. GATHER SOURCE REAL IMAGES
source_real_dirs = [
    "/mnt/ai-storage/aigc_data/datasets/portrait_remediation/div2k_extracted",
    "/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_headshot",
    "/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_portrait",
    "/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_retouched",
    "/mnt/ai-storage/aigc_data/datasets/portrait_remediation/real_studio",
    "/mnt/ai-storage/aigc_data/datasets/remediation_expansion_pool/ImageNet_Authentic_Photo",
    "/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real"
]

all_source_files = []
for d in source_real_dirs:
    if os.path.exists(d):
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_source_files.append(os.path.join(root, f))

random.shuffle(all_source_files)
print(f"Total Available Real Source Images: {len(all_source_files):,}")

# 3. GENERATE TARGETED HARD NEGATIVES (1,000 PER CATEGORY = 8,000 TOTAL)
target_per_category = 1000
provenance_manifest = []

print(f"Generating {len(TRANSFORMS)} categories x {target_per_category} = {len(TRANSFORMS)*target_per_category:,} Paired Hard Negatives...")

file_idx = 0
for cat_name, fn in TRANSFORMS.items():
    cat_dir = os.path.join(OUTPUT_DIR, cat_name)
    generated = 0
    t0 = time.time()
    
    while generated < target_per_category and file_idx < len(all_source_files):
        src_path = all_source_files[file_idx]
        file_idx += 1
        
        try:
            with Image.open(src_path) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                
                # Apply transformation
                transformed = fn(img_rgb)
                
                # Save transformed paired image
                base_name = f"{cat_name}_{generated:05d}.jpg"
                out_path = os.path.join(cat_dir, base_name)
                transformed.save(out_path, format="JPEG", quality=90)
                
                provenance_manifest.append({
                    "original_path": src_path,
                    "transformed_path": out_path,
                    "transformation_category": cat_name,
                    "original_resolution": [w, h],
                    "label": 0, # Guaranteed Real Ground Truth
                    "provenance": "PILLAR1_PAIRED_HARD_NEGATIVE"
                })
                generated += 1
        except Exception as e:
            continue
            
    elapsed = time.time() - t0
    print(f"  ✓ {cat_name:30s}: {generated:4d} images generated ({elapsed:.1f}s)")

# Save Master Provenance Manifest
manifest_path = "/home/manan/aigc_robust_detection/reports/paired_hard_negatives_manifest.json"
with open(manifest_path, "w") as f:
    json.dump({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_hard_negatives": len(provenance_manifest),
        "categories": list(TRANSFORMS.keys()),
        "samples": provenance_manifest
    }, f, indent=2)

print("=" * 85)
print(f"  PILLAR 1 COMPLETE: {len(provenance_manifest):,} Paired Hard Negatives Generated")
print(f"  Manifest Saved : {manifest_path}")
print("=" * 85)
