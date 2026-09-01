#!/usr/bin/env python3
"""
production_master_engine.py
---------------------------
Final Production Inference Engine for the Master Intelligent Forensic Model.
Supports FP32, FP16, and FP8/INT8 models.

Independently outputs:
  - 3-Way Verdict: REAL / PARTIAL_AIGC / FULL_AIGC
  - Calibrated Confidence
  - Spatial Localization Heatmap Overlay
  - Suspicious Region Bounding Boxes
  - Affected Area Percentage (0% - 100%)
  - Evidence Breakdown from all 11 specialist sources

Usage:
  >>> from scripts.final.production_master_engine import ProductionMasterEngine
  >>> engine = ProductionMasterEngine(precision="FP16", device="cuda:0")
  >>> result = engine.predict("/path/to/image.jpg", save_heatmap=True)
  >>> print(result["verdict"], result["confidence"], result["affected_area_percentage"])
"""

import os
import sys
import time
from typing import Dict, Any, Union, Optional, List
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.package_three_production_models import CompleteMasterIntelligentForensicModel

class ProductionMasterEngine:
    def __init__(
        self,
        precision: str = "FP16",
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.precision = precision.upper()
        if device is None:
            if self.precision in ["FP16", "FP8_INT8", "FP8"] and torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        base_dir = "/home/manan/aigc_robust_detection/checkpoints/production_candidate"
        if checkpoint_path is None:
            if self.precision in ["FP8", "FP8_INT8"]:
                checkpoint_path = f"{base_dir}/master_intelligent_forensic_model_fp8.pt"
            elif self.precision == "FP16":
                checkpoint_path = f"{base_dir}/master_intelligent_forensic_model_fp16.pt"
            else:
                checkpoint_path = f"{base_dir}/master_intelligent_forensic_model_fp32.pt"

        self.checkpoint_path = checkpoint_path
        self.dtype = torch.float16 if self.precision in ["FP16", "FP8", "FP8_INT8"] else torch.float32

        print(f"🚀 Initializing Production Master Forensic Engine [{self.precision}] on {self.device}...")
        t0 = time.time()

        self.model = CompleteMasterIntelligentForensicModel()
        if self.precision in ["FP16", "FP8", "FP8_INT8"]:
            self.model = self.model.half()

        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"]

        # Dequantize if FP8/INT8
        if self.precision in ["FP8", "FP8_INT8"]:
            clean_sd = {}
            for k, v in sd.items():
                if isinstance(v, dict) and v.get("is_quantized", False):
                    clean_sd[k] = (v["qweight"].float() * v["scale"]).to(self.dtype)
                else:
                    clean_sd[k] = v.to(self.dtype) if isinstance(v, torch.Tensor) else v
            self.model.load_state_dict(clean_sd)
        else:
            self.model.load_state_dict(sd)

        self.model = self.model.to(self.device).eval()
        self.total_parameters = ckpt.get("total_parameters", 1820886125)
        self.load_time_seconds = time.time() - t0
        print(f"✅ Production Master Model Loaded in {self.load_time_seconds:.2f}s | Parameters: {self.total_parameters:,}")

        # Transforms
        self.t_224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.t_256 = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.t_384 = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        save_heatmap: bool = True,
        heatmap_out_dir: str = "/home/manan/aigc_robust_detection/reports/production_heatmaps"
    ) -> Dict[str, Any]:
        t_start = time.time()
        
        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            img = Image.open(image_input).convert("RGB")
            stem = Path(image_input).stem
        else:
            img = image_input.convert("RGB")
            img_path_str = "in_memory_image.jpg"
            stem = "in_memory_image"

        orig_w, orig_h = img.size
        img_np = np.array(img)

        # Multi-Modal Tensor Preparation
        img_224 = self.t_224(img).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_256_5v = self.t_256(img).unsqueeze(0).unsqueeze(1).repeat(1, 5, 1, 1, 1).to(self.device, dtype=self.dtype)
        img_384 = self.t_384(img).unsqueeze(0).to(self.device, dtype=self.dtype)
        srm_feats = self.model.specialists.v3_c0_champion.srm_proj[0].weight.new_zeros((1, 36)).to(self.device, dtype=self.dtype)

        # Hierarchical Multi-Scale Patch Preparation (512, 768, 1024)
        patch_tensors = [img_224.squeeze(0)]
        patch_coords = [[0.0, 0.0, 1.0, 1.0, 1.0]]

        for scale in [512, 768]:
            if orig_w >= scale and orig_h >= scale:
                step = scale // 2
                for y in range(0, orig_h - scale + 1, step):
                    for x in range(0, orig_w - scale + 1, step):
                        crop = img.crop((x, y, x + scale, y + scale))
                        patch_tensors.append(self.t_224(crop).to(self.device, dtype=self.dtype))
                        patch_coords.append([x / orig_w, y / orig_h, (x + scale) / orig_w, (y + scale) / orig_h, scale / max(orig_w, orig_h)])
                        if len(patch_tensors) >= 16:
                            break
                    if len(patch_tensors) >= 16:
                        break

        patch_tensors_t = torch.stack(patch_tensors)
        patch_coords_t = torch.tensor(patch_coords, dtype=self.dtype, device=self.device)

        # Forward Pass
        out = self.model(img_224, img_256_5v, img_384, srm_feats, patch_tensors_t, patch_coords_t)
        latency_ms = (time.time() - t_start) * 1000

        probs = out["calibrated_probs"].float().cpu().squeeze(0)
        p_real = float(probs[0].item())
        p_partial = float(probs[1].item())
        p_full = float(probs[2].item())

        class_idx = int(probs.argmax().item())
        class_labels = ["REAL", "PARTIAL_AIGC", "FULL_AIGC"]
        verdict = class_labels[class_idx]
        confidence = float(probs[class_idx].item())

        # Anomaly Map & Spatial Segmentation
        seg_mask = out["segmentation_heatmap"].float().cpu().squeeze().numpy() # (64, 64)
        seg_mask_resized = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # Calculate Affected Area %
        threshold = 0.50
        affected_pixels = np.sum(seg_mask_resized > threshold)
        total_pixels = orig_w * orig_h
        affected_area_pct = float((affected_pixels / total_pixels) * 100.0) if verdict != "REAL" else 0.0

        # Extract Suspicious Bounding Boxes
        suspicious_boxes = []
        if verdict != "REAL":
            binary_mask = (seg_mask_resized > threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1000: # Minimum significant area
                    x, y, w, h = cv2.boundingRect(cnt)
                    suspicious_boxes.append({
                        "box_2d": [int(y), int(x), int(y + h), int(x + w)],
                        "confidence": round(float(np.mean(seg_mask_resized[y:y+h, x:x+w])), 4),
                        "label": "Manipulated_AIGC_Region"
                    })

        # Generate & Save Visual Heatmap Overlay
        heatmap_path = None
        if save_heatmap:
            os.makedirs(heatmap_out_dir, exist_ok=True)
            heatmap_path = os.path.join(heatmap_out_dir, f"{stem}_{self.precision.lower()}_master_heatmap.jpg")
            norm_map = np.uint8(255 * np.clip(seg_mask_resized, 0, 1))
            color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_bgr, 0.60, color_map, 0.40, 0)
            
            # Draw Bounding Boxes
            for box in suspicious_boxes:
                y1, x1, y2, x2 = box["box_2d"]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(overlay, f"AI: {box['confidence']:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(heatmap_path, overlay)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "probabilities": {
                "real": round(p_real, 4),
                "partial_ai": round(p_partial, 4),
                "full_aigc": round(p_full, 4)
            },
            "affected_area_percentage": round(affected_area_pct, 2),
            "suspicious_regions_count": len(suspicious_boxes),
            "suspicious_regions": suspicious_boxes,
            "heatmap_path": heatmap_path,
            "evidence_breakdown": {
                "V2_AIDE_Spectral_Score": round(float(out["v2_spectral_score"].item()), 4),
                "V3_Ensemble_Gated_Score": round(float(out["v3_gated_score"].item()), 4),
                "Specialist_Logits": dict((k, round(float(v.item()), 4)) for k, v in out["specialist_logits"].items())
            },
            "runtime_telemetry": {
                "precision": self.precision,
                "device": str(self.device),
                "latency_ms": round(latency_ms, 2),
                "total_parameters": self.total_parameters
            }
        }
