#!/usr/bin/env python3
"""
predict.py
--------------------------------------------------------------------------------
Official Hackathon Submission Evaluation Script for Track 5 (TikTok TechJam)
Takes an input directory of images and outputs a standardized JSON file
containing 'image_path' and 'pred' (AIGC probability score in [0.0, 1.0]).

Usage:
    python predict.py --input-dir /path/to/images --output predictions.json
    python predict.py --input-dir ./test_inputs --checkpoint checkpoints/distilled/highcap_distilled_forensic_model_int8.pt
--------------------------------------------------------------------------------
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torchvision.transforms as T
from PIL import Image

# Import Standalone Student Model Architecture & Production Wrapper
try:
    from scripts.final.highcap_distilled_forensic_model import HighCapacityDistilledForensicModel, HighCapacityStudentForensicModel
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from scripts.final.highcap_distilled_forensic_model import HighCapacityDistilledForensicModel, HighCapacityStudentForensicModel


def get_image_paths(directory: str) -> List[Path]:
    """Recursively discover all valid image files in a directory."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".tiff"}
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    
    files = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            ext = Path(f).suffix.lower()
            if ext in valid_exts and not f.startswith("._") and not f.startswith("."):
                files.append(Path(root) / f)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="ENTROPY: Track 5 Batch AIGC Image Directory Predictor")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to input directory containing images")
    parser.add_argument("--output", type=str, default="predictions.json", help="Path to output JSON file (default: predictions.json)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "FP32", "FP16", "INT8"], help="Model precision")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda', 'mps', or 'cpu'")
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"🚀 Initializing ENTROPY Forensic Inference on device: {device}")

    # Determine checkpoint path
    checkpoint_candidates = [
        args.checkpoint,
        "checkpoints/distilled/highcap_distilled_forensic_model_int8.pt",
        "checkpoints/distilled/highcap_distilled_forensic_model_fp16.pt",
        "checkpoints/distilled/highcap_distilled_forensic_model_fp32.pt",
        "checkpoints/distilled/master_distilled_forensic_model_fp16.pt",
        "checkpoints/distilled/master_distilled_forensic_model_int8.pt",
        "/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_int8.pt",
    ]

    ckpt_path = None
    for cand in checkpoint_candidates:
        if cand and os.path.exists(cand):
            ckpt_path = cand
            break

    if not ckpt_path:
        print(f"❌ Error: No valid checkpoint found. Please specify --checkpoint path.")
        sys.exit(1)

    print(f"📦 Loading Checkpoint: {ckpt_path}")

    # Auto-detect precision from filename if not specified
    precision = args.precision.upper()
    if precision == "AUTO":
        if "int8" in ckpt_path.lower():
            precision = "INT8"
        elif "fp16" in ckpt_path.lower():
            precision = "FP16"
        else:
            precision = "FP32"

    # Initialize production wrapper
    engine = HighCapacityDistilledForensicModel(
        checkpoint_path=ckpt_path,
        precision=precision,
        device=device
    )

    # Discover images
    image_paths = get_image_paths(args.input_dir)
    print(f"🖼️ Found {len(image_paths)} images in '{args.input_dir}'")

    if len(image_paths) == 0:
        print("No images found. Writing empty predictions array.")
        with open(args.output, "w") as f:
            json.dump([], f, indent=2)
        return

    results = []
    for idx, img_path in enumerate(image_paths, 1):
        try:
            res = engine.predict(img_path, save_heatmap=False)
            ai_score = float(res["ai_probability"])
            results.append({
                "image_path": str(img_path),
                "pred": round(ai_score, 4)
            })
        except Exception as e:
            print(f"⚠️ Error evaluating {img_path}: {e}", file=sys.stderr)
            results.append({
                "image_path": str(img_path),
                "pred": 0.5
            })

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"  Processed {idx}/{len(image_paths)} images...")

    # Write output JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Prediction complete! Evaluated {len(results)} images.")
    print(f"📄 Results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
