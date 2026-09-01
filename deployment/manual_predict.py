#!/usr/bin/env python3
"""
deployment/manual_predict.py
Lightweight Standalone Interactive Manual Inference Tester for Local Mac Execution.
Zero heavy research/training dependencies. Reconstructs and runs ScientificVisionDetector-ConfigA.
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
from PIL import Image, ImageOps
import numpy as np
import torch

# Ensure deployment directory is in python path
DEPLOY_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.portable_model import (
    load_portable_champion_model,
    portable_eval_transform,
    get_preferred_device
)

DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "production" / "final_champion_frozen_model.pt"

# Production Operational Thresholds
THRESHOLDS = {
    "standard": (0.500000, "Standard 50% Boundary"),
    "low_fpr_10": (0.726040, "FPR <= 1.00% High-Throughput Gate"),
    "low_fpr_05": (0.931236, "FPR <= 0.50% Moderation Gate"),
    "low_fpr_01": (0.984399, "FPR <= 0.10% Enterprise Gate (Recommended)"),
    "low_fpr_005": (0.990601, "FPR <= 0.05% High-Security Gate"),
    "low_fpr_001": (0.994351, "FPR <= 0.01% Zero-False-Alarm Gate")
}

def load_and_preprocess_image(image_path: Union[str, Path]) -> Tuple[torch.Tensor, Image.Image]:
    """Loads and standardizes input image for evaluation."""
    p = Path(image_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
        
    with Image.open(p) as raw_img:
        img = ImageOps.exif_transpose(raw_img).convert("RGB")
        tensor = portable_eval_transform(img).unsqueeze(0)
    return tensor, img

def run_single_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    temperature: float = 1.5230212761606914,
    gate_mode: str = "low_fpr_01"
) -> Dict[str, Any]:
    """Runs a single forward pass and returns calibrated probability and forensic signals."""
    thresh_val, thresh_desc = THRESHOLDS.get(gate_mode, THRESHOLDS["low_fpr_01"])
    
    t0 = time.perf_counter()
    tensor_dev = tensor.to(device)
    with torch.inference_mode():
        # Compute logits and auxiliary evidence
        res = model(tensor_dev, return_evidence=True)
        if isinstance(res, tuple):
            logit_t, ev_t, srm_t = res
            raw_logit = float(logit_t.cpu().item())
            srm_energy = float(srm_t.abs().mean().cpu().item())
            ev_scores = ev_t.squeeze(0).cpu().numpy().tolist()
        else:
            raw_logit = float(res.cpu().item())
            srm_energy = 0.0
            ev_scores = []
            
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    # Apply Temperature Scaling
    calibrated_logit = raw_logit / temperature
    p_aigc = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
    is_aigc = p_aigc >= thresh_val
    pred_label = "AIGC_SYNTHETIC" if is_aigc else "AUTHENTIC_REAL"
    
    return {
        "raw_logit": raw_logit,
        "calibrated_probability": p_aigc,
        "prediction": pred_label,
        "is_aigc": is_aigc,
        "threshold": thresh_val,
        "threshold_description": thresh_desc,
        "temperature": temperature,
        "latency_ms": latency_ms,
        "srm_residual_energy": srm_energy,
        "evidence_scores": ev_scores
    }

def print_result_block(
    image_path: str,
    device: torch.device,
    res: Dict[str, Any]
):
    """Prints the standardized output format."""
    p_aigc = res["calibrated_probability"]
    p_pct = p_aigc * 100.0
    
    print("\n" + "=" * 50)
    print("IMAGE")
    print("=" * 50)
    print(f"Path:                     {image_path}")
    print(f"Device:                   {device}")
    print(f"P(AIGC):                  {p_aigc:.6f} ({p_pct:.2f}%)")
    print(f"Prediction:               {res['prediction']}")
    print(f"Threshold:                {res['threshold']:.6f} ({res['threshold_description']})")
    print(f"Calibration Temperature:  {res['temperature']:.6f}")
    print(f"Latency:                  {res['latency_ms']:.2f} ms")
    
    # Forensic Auxiliary Signals (Separated from Primary Classification)
    if res.get("srm_residual_energy", 0.0) > 0.0:
        print("\n[Forensic Auxiliary Signals]")
        print(f"  - SRM Wavelet Residual Energy: {res['srm_residual_energy']:.4f}")
        status = "ANOMALOUS_HIGH_FREQUENCY" if res["srm_residual_energy"] > 2.5 else "CONSISTENT_NATURAL"
        print(f"  - Frequency Status:            {status}")
    print("=" * 50 + "\n")

def main():
    print("=" * 70)
    print("  Final AIGC Detector — Standalone Local Inference")
    print("=" * 70)
    
    ckpt_path = DEFAULT_CHECKPOINT_PATH
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        ckpt_path = Path(sys.argv[1])
        
    device = get_preferred_device()
    print(f"Target Device: {device} ({'Apple Silicon Metal Performance Shaders' if device.type == 'mps' else 'CPU Standard'})")
    print(f"Loading Frozen Checkpoint: {ckpt_path}...")
    
    t_load_0 = time.perf_counter()
    try:
        model, metadata = load_portable_champion_model(ckpt_path, device=device)
    except Exception as e:
        print(f"\n[FATAL ERROR] Model instantiation failed: {e}")
        sys.exit(1)
        
    load_time_s = time.perf_counter() - t_load_0
    
    print(f"Model SHA-256:     {metadata['file_sha256']}")
    print(f"Parameter Hash:    {metadata['parameter_hash']}")
    print(f"Total Parameters:  {metadata['total_parameters']:,}")
    print(f"Trainable Params:  {metadata['trainable_parameters']:,} ({metadata['trainable_parameters']/metadata['total_parameters']*100:.2f}%)")
    print(f"Load Time:         {load_time_s:.2f} s")
    print("\nMODEL LOAD SUCCESS\n")
    
    temperature = metadata.get("temperature", 1.5230212761606914)
    
    # Interactive CLI Loop
    while True:
        try:
            user_input = input("Enter image path (or 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
            
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break
            
        # Clean quote marks if pasted from terminal/finder
        cleaned_path = user_input.strip("'\"")
        
        try:
            tensor, _ = load_and_preprocess_image(cleaned_path)
            res = run_single_inference(
                model=model,
                tensor=tensor,
                device=device,
                temperature=temperature,
                gate_mode="low_fpr_01"
            )
            print_result_block(cleaned_path, device, res)
        except Exception as e:
            print(f"\n[ERROR] Inference failed on '{cleaned_path}': {e}\n")

if __name__ == "__main__":
    main()
