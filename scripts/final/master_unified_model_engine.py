#!/usr/bin/env python3
"""
master_unified_model_engine.py
------------------------------
Production Inference Engine for the Compiled Master Monolithic Forensic Model.
Provides 1-line instantiation, load, and full-spectrum forensic evaluation.

Usage Example:
  >>> from scripts.final.master_unified_model_engine import MasterUnifiedForensicEngine
  >>> engine = MasterUnifiedForensicEngine(precision="FP16", device="cuda:0")
  >>> result = engine.predict("/path/to/image.jpg")
  >>> print(result["verdict"], result["confidence"])
"""

import os
import sys
import time
from typing import Dict, Any, Union, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.compile_master_unified_model import MasterUnifiedForensicModel

class MasterUnifiedForensicEngine:
    def __init__(
        self,
        precision: str = "FP16",
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initializes the compiled master monolithic forensic model.
        
        Args:
            precision (str): "FP16" (default, 3.39 GB) or "FP32" (6.78 GB).
            device (str): "cuda:0", "cpu", or auto-detect.
            checkpoint_path (str): Custom path to .pt checkpoint if omitted.
        """
        self.precision = precision.upper()
        if device is None:
            if self.precision == "FP16" and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            if self.precision == "FP16":
                checkpoint_path = "/home/manan/aigc_robust_detection/checkpoints/compiled/master_unified_forensic_model_fp16.pt"
            else:
                checkpoint_path = "/home/manan/aigc_robust_detection/checkpoints/compiled/master_unified_forensic_model_fp32.pt"

        self.checkpoint_path = checkpoint_path
        self.dtype = torch.float16 if self.precision == "FP16" else torch.float32

        print(f"🚀 Initializing Master Unified Forensic Engine [{self.precision}] on {self.device}...")
        t0 = time.time()

        # Instantiate Architecture
        self.model = MasterUnifiedForensicModel()
        if self.precision == "FP16":
            self.model = self.model.half()

        # Load Compiled Monolith State Dict
        ckpt_data = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(ckpt_data["model_state_dict"])
        self.model = self.model.to(self.device).eval()

        self.total_parameters = ckpt_data.get("total_parameters", 1818496169)
        self.load_time_seconds = time.time() - t0
        print(f"✅ Master Model Loaded in {self.load_time_seconds:.2f}s | Parameters: {self.total_parameters:,}")

        # Transforms
        self.t_224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.t_256 = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.t_384 = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    @torch.no_grad()
    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Executes end-to-end forensic inference on a target image.
        
        Returns:
            Dict containing verdict, confidence, probabilities, specialist breakdown, and latency.
        """
        t_start = time.time()
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        # Multi-Modal Tensor Preparation
        img_224 = self.t_224(img).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_256_5v = self.t_256(img).unsqueeze(0).unsqueeze(1).repeat(1, 5, 1, 1, 1).to(self.device, dtype=self.dtype)
        img_384 = self.t_384(img).unsqueeze(0).to(self.device, dtype=self.dtype)
        srm_feats = self.model.v3_c0_champion.srm_proj[0].weight.new_zeros((1, 36)).to(self.device, dtype=self.dtype)

        patch_tensors = [img_224.squeeze(0)]
        patch_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=self.dtype, device=self.device)
        patch_tensors_t = torch.stack(patch_tensors).to(self.device, dtype=self.dtype)

        # Monolithic Forward Pass
        out = self.model(img_224, img_256_5v, img_384, srm_feats, patch_tensors_t, patch_coords)
        latency_ms = (time.time() - t_start) * 1000

        ai_prob = float(out["fused_ai_probability"].item())
        real_prob = float(out["real_probability"].item())
        partial_prob = float(out["partial_ai_probability"].item())
        full_prob = float(out["full_aigc_probability"].item())

        # Determine 3-Way Verdict
        if ai_prob < 0.35 and partial_prob < 0.40:
            verdict = "REAL"
            confidence = real_prob
        elif partial_prob > 0.45 and full_prob < 0.60:
            verdict = "PARTIAL_AIGC"
            confidence = partial_prob
        else:
            verdict = "FULL_AIGC"
            confidence = max(ai_prob, full_prob)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "probabilities": {
                "real": round(real_prob, 4),
                "partial_ai": round(partial_prob, 4),
                "full_aigc": round(full_prob, 4),
                "fused_ai": round(ai_prob, 4)
            },
            "evidence": {
                "v2_spectral_score": round(float(out["v2_spectral_score"].item()), 4),
                "v3_gated_score": round(float(out["v3_gated_score"].item()), 4),
                "v5_spatial_score": round(float(out["v5_spatial_score"].item()), 4),
                "specialist_logits": dict((k, round(float(v.item()), 4)) for k, v in out["specialist_logits"].items())
            },
            "runtime": {
                "latency_ms": round(latency_ms, 2),
                "precision": self.precision,
                "device": str(self.device),
                "total_parameters": self.total_parameters
            }
        }
