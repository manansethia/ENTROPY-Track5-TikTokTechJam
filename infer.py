#!/usr/bin/env python3
"""
infer.py — Authoritative Hackathon Inference Interface
Evaluates the production Triple-Hybrid Champion Model on a directory of images and emits standard predictions JSON.

Usage:
    python infer.py --input-dir ./test_images --output predictions.json
    python infer.py --input-dir ./test_images --output predictions.json --device auto --batch-size 16
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.forensic_adapter import get_forensic_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="AIGC Robust Detection — Production Triple-Hybrid Inference CLI"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Path to directory containing input evaluation images (JPG, PNG, WEBP, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="predictions.json",
        help="Destination path for predictions JSON file"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Device to use for inference ('auto', 'cuda', 'mps', 'cpu')"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of images to process"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include full forensic metadata, EXIF, and provenance breakdown"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
    image_paths = sorted([
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ])

    if not image_paths:
        print(f"[WARNING] No valid image files found in {input_dir}", file=sys.stderr)
        with open(args.output, "w") as f:
            json.dump([], f, indent=2)
        return

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"[AIGC Forensics] Found {len(image_paths)} images to analyze.")
    print(f"[AIGC Forensics] Initializing Forensic Model Adapter...")

    adapter = get_forensic_adapter()
    meta = adapter.get_metadata()
    print(f"[AIGC Forensics] Model: {meta['model_name']} ({meta['architecture']})")
    print(f"[AIGC Forensics] Device: {meta['operating_device']} | Parameters: {meta['parameter_count']:,}")

    t0 = time.time()
    results = []

    for idx, img_path in enumerate(image_paths, 1):
        try:
            rel_path = str(img_path.relative_to(input_dir))
        except ValueError:
            rel_path = str(img_path)

        res = adapter.predict(str(img_path), filename=img_path.name)
        prob = res.get("probability_aigc", 0.5)

        if args.detailed:
            results.append({
                "image_path": rel_path,
                "pred": round(prob, 4),
                "verdict": res.get("verdict"),
                "confidence": res.get("confidence"),
                "affected_area_percentage": res.get("affected_area_percentage"),
                "fft_high_frequency_ratio": res["spatial_forensics"]["fft_high_frequency_ratio"],
                "srm_residual_energy": res["spatial_forensics"]["srm_residual_energy"],
                "sha256": res["file_info"]["sha256"]
            })
        else:
            # Standard Hackathon Required Output Format
            results.append({
                "image_path": rel_path,
                "pred": round(prob, 4)
            })

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] Processed: {img_path.name} -> P(AIGC)={prob:.4f}")

    elapsed = time.time() - t0
    fps = len(image_paths) / max(0.001, elapsed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Completed inference across {len(image_paths)} images in {elapsed:.2f}s ({fps:.1f} FPS).")
    print(f"[SUCCESS] Predictions saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
