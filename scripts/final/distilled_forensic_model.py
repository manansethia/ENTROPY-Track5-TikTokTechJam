#!/usr/bin/env python3
"""
distilled_forensic_model.py
---------------------------
True Single-Model Distilled Forensic Neural Network.
Contains:
  1. Spatial Backbone (ConvNeXt multi-scale feature extractor)
  2. Integrated SRM & Spectral Residual Filter Bank (30 5x5 filters)
  3. Feature Pyramid Fusion (FPN) combining spatial and spectral semantics
  4. 3-Way Classification Head: [P(Real), P(Partial-AI), P(Full-AIGC)]
  5. High-Resolution Spatial Heatmap Decoder (64x64 continuous anomaly map)

At inference time, this file is 100% standalone:
  NO V2, NO V3, NO C0-C7, NO V5 dependencies.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np

# -------------------------------------------------------------------------
# High-Pass SRM Spectral Filter Bank Layer
# -------------------------------------------------------------------------
class SRMSpectralLayer(nn.Module):
    """
    Learned/Initialized 30-filter High-Pass SRM Filter Bank for capturing
    micro-textures, upscaling artifacts, and diffusion frequency residuals.
    """
    def __init__(self, in_channels: int = 3, num_filters: int = 30):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm = nn.BatchNorm2d(num_filters)
        self.act = nn.Tanh()

        # Initialize with standard high-pass residual filter shapes
        with torch.no_grad():
            w = torch.zeros(num_filters, in_channels, 5, 5)
            for i in range(num_filters):
                # Center-surround high-pass pattern
                w[i, :, 2, 2] = -1.0
                w[i, :, 1:4, 1:4] += 1.0 / 8.0
            self.conv.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))

# -------------------------------------------------------------------------
# Single Distilled Student Forensic Neural Network
# -------------------------------------------------------------------------
class SingleStudentForensicModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        embed_dim: int = 384,
        heatmap_res: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        self.heatmap_res = heatmap_res

        # 1. Visual Feature Stem & Stages (ConvNeXt-style building blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            nn.LayerNorm([96, 56, 56], eps=1e-6)
        )

        # Stage 1 (56x56, 96D)
        self.stage1 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=7, padding=3, groups=96),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            nn.GELU()
        )

        # Downsample 1 -> Stage 2 (28x28, 192D)
        self.down1 = nn.Sequential(nn.BatchNorm2d(96), nn.Conv2d(96, 192, kernel_size=2, stride=2))
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=7, padding=3, groups=192),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.BatchNorm2d(192),
            nn.GELU()
        )

        # Downsample 2 -> Stage 3 (14x14, 384D)
        self.down2 = nn.Sequential(nn.BatchNorm2d(192), nn.Conv2d(192, embed_dim, kernel_size=2, stride=2))
        self.stage3 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=7, padding=3, groups=embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Downsample 3 -> Stage 4 (7x7, 768D)
        self.down3 = nn.Sequential(nn.BatchNorm2d(embed_dim), nn.Conv2d(embed_dim, 768, kernel_size=2, stride=2))
        self.stage4 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=7, padding=3, groups=768),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Conv2d(768, 768, kernel_size=1),
            nn.BatchNorm2d(768),
            nn.GELU()
        )

        # 2. High-Pass Spectral Branch (SRM + Frequency Energy)
        self.srm_branch = SRMSpectralLayer(in_channels=3, num_filters=32)
        self.srm_encoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=4, padding=1), # 14x14
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # 3. Cross-Modal Fusion & Feature Pyramid
        self.cross_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # 4. Global Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(768 + embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes)
        )

        # 5. Spatial Heatmap / Localization Decoder
        self.heatmap_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), # 28x28
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Upsample(size=(heatmap_res, heatmap_res), mode="bilinear", align_corners=False), # 64x64
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Visual stages
        s0 = self.stem(x)       # (B, 96, 56, 56)
        s1 = self.stage1(s0)    # (B, 96, 56, 56)
        s2 = self.stage2(self.down1(s1)) # (B, 192, 28, 28)
        s3 = self.stage3(self.down2(s2)) # (B, 384, 14, 14)
        s4 = self.stage4(self.down3(s3)) # (B, 768, 7, 7)

        # Spectral branch
        srm_out = self.srm_branch(x)
        srm_feat = self.srm_encoder(srm_out) # (B, 384, 14, 14)

        # Fused intermediate representation
        fused_14 = self.cross_fuse(torch.cat([s3, srm_feat], dim=1)) # (B, 384, 14, 14)

        # Global features
        vis_global = self.global_pool(s4).flatten(1)       # (B, 768)
        spec_global = self.global_pool(fused_14).flatten(1) # (B, 384)
        joint_feat = torch.cat([vis_global, spec_global], dim=1) # (B, 1152)

        # Classification logits & probabilities
        class_logits = self.classifier(joint_feat) # (B, 3)
        probs = F.softmax(class_logits, dim=-1)

        # Heatmap / Segmentation mask
        heatmap = self.heatmap_decoder(fused_14) # (B, 1, 64, 64)

        return {
            "class_logits": class_logits,
            "probabilities": probs,
            "real_probability": probs[:, 0],
            "partial_ai_probability": probs[:, 1],
            "full_aigc_probability": probs[:, 2],
            "ai_probability": 1.0 - probs[:, 0],
            "segmentation_heatmap": heatmap,
            "joint_features": joint_feat
        }

# -------------------------------------------------------------------------
# Production Engine Wrapper for Standalone Distilled Student Model
# -------------------------------------------------------------------------
class DistilledForensicModel:
    """
    1-Line Standalone Production Inference Interface.
    Loads ONLY the single student model checkpoint.
    Zero teacher dependencies.
    """
    def __init__(
        self,
        checkpoint_path: str,
        precision: str = "FP16",
        device: Optional[str] = None
    ):
        self.precision = precision.upper()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.dtype = torch.float16 if self.precision in ["FP16", "FP8", "INT8"] and "cuda" in str(self.device) else torch.float32

        t0 = time.time()
        self.model = SingleStudentForensicModel()
        
        # Load weights
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Handle INT8 dequantization if saved as quantized dict
        if self.precision in ["INT8", "FP8"]:
            clean_sd = {}
            for k, v in sd.items():
                if isinstance(v, dict) and v.get("is_quantized", False):
                    clean_sd[k] = (v["qweight"].float() * v["scale"]).to(self.dtype)
                else:
                    clean_sd[k] = v.to(self.dtype) if isinstance(v, torch.Tensor) else v
            self.model.load_state_dict(clean_sd)
        else:
            self.model.load_state_dict(sd)

        if self.precision in ["FP16", "FP8", "INT8"] and "cuda" in str(self.device):
            self.model = self.model.half()

        self.model = self.model.to(self.device).eval()
        self.total_parameters = sum(p.numel() for p in self.model.parameters())
        self.load_time = time.time() - t0

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @classmethod
    def load(
        cls,
        checkpoint_path: str = "/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_fp16.pt",
        precision: str = "FP16",
        device: Optional[str] = None
    ) -> "DistilledForensicModel":
        return cls(checkpoint_path=checkpoint_path, precision=precision, device=device)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image],
        save_heatmap: bool = True,
        heatmap_out_dir: str = "/home/manan/aigc_robust_detection/reports/production_heatmaps"
    ) -> Dict[str, Any]:
        t_start = time.time()

        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
            stem = Path(image_input).stem
        else:
            img = image_input.convert("RGB")
            stem = "in_memory_image"

        orig_w, orig_h = img.size
        img_np = np.array(img)

        # Single tensor forward pass
        img_t = self.transform(img).unsqueeze(0).to(self.device, dtype=self.dtype)
        out = self.model(img_t)
        latency_ms = (time.time() - t_start) * 1000

        # Parse probabilities
        probs = out["probabilities"].float().cpu().squeeze(0)
        p_real = float(probs[0].item())
        p_partial = float(probs[1].item())
        p_full = float(probs[2].item())

        class_idx = int(probs.argmax().item())
        class_labels = ["REAL", "PARTIAL_AIGC", "FULL_AIGC"]
        verdict = class_labels[class_idx]
        confidence = float(probs[class_idx].item())

        # Resize heatmap
        seg_mask = out["segmentation_heatmap"].float().cpu().squeeze().numpy() # (64, 64)
        seg_mask_resized = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        threshold = 0.50
        affected_pixels = np.sum(seg_mask_resized > threshold)
        total_pixels = orig_w * orig_h
        affected_area_pct = float((affected_pixels / total_pixels) * 100.0) if verdict != "REAL" else 0.0

        # Extract suspicious bounding boxes
        suspicious_boxes = []
        if verdict != "REAL":
            binary_mask = (seg_mask_resized > threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    suspicious_boxes.append({
                        "box_2d": [int(y), int(x), int(y + h), int(x + w)],
                        "confidence": round(float(np.mean(seg_mask_resized[y:y+h, x:x+w])), 4),
                        "label": "Manipulated_AIGC_Region"
                    })

        heatmap_path = None
        if save_heatmap:
            os.makedirs(heatmap_out_dir, exist_ok=True)
            heatmap_path = os.path.join(heatmap_out_dir, f"{stem}_distilled_{self.precision.lower()}_heatmap.jpg")
            norm_map = np.uint8(255 * np.clip(seg_mask_resized, 0, 1))
            color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_bgr, 0.60, color_map, 0.40, 0)
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
            "ai_probability": round(1.0 - p_real, 4),
            "affected_area_percentage": round(affected_area_pct, 2),
            "suspicious_regions_count": len(suspicious_boxes),
            "suspicious_regions": suspicious_boxes,
            "heatmap_path": heatmap_path,
            "runtime_telemetry": {
                "architecture": "SingleStudentForensicModel",
                "total_parameters": self.total_parameters,
                "precision": self.precision,
                "device": str(self.device),
                "single_image_latency_ms": round(latency_ms, 2)
            }
        }
