#!/usr/bin/env python3
"""
master_fused_forensic_engine.py
-------------------------------
Production Standalone Master Forensic Inference Engine.

Delivers deterministic, autonomous 3-way verdicts:
  1. REAL / AUTHENTIC
  2. PARTIAL_AIGC / LOCALIZED MANIPULATION (with bounding boxes & % affected area)
  3. FULL_AIGC / SYNTHETIC GENERATION

Requires 0 external API calls or human-in-the-loop review.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

V5_CHECKPOINT = "/home/manan/aigc_robust_detection/checkpoints/experimental/v5/v5_champion_cag.pt"
V3_CHECKPOINT = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v3.pt"
V2_CHECKPOINT = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v2.pt"

class V5CAGModel(nn.Module):
    def __init__(self, feature_dim=768, pos_dim=128, fused_dim=256):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(5, pos_dim),
            nn.LayerNorm(pos_dim),
            nn.GELU(),
            nn.Linear(pos_dim, pos_dim)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + pos_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU()
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.whole_classifier = nn.Linear(fused_dim, 3)
        self.patch_classifier = nn.Linear(fused_dim, 1)
        self.seg_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 64),
            nn.Sigmoid()
        )

    def forward(self, g_feat: torch.Tensor, p_feats: torch.Tensor, p_coords: torch.Tensor):
        N = p_feats.shape[0]
        g_rep = g_feat.expand(N, -1)
        pos_emb = self.pos_mlp(p_coords)
        combined = torch.cat([g_rep, p_feats, pos_emb], dim=-1)
        fused = self.fusion_mlp(combined)
        patch_logits = self.patch_classifier(fused).squeeze(-1)
        attn_scores = self.attention_gate(fused)
        attn_weights = F.softmax(attn_scores, dim=0)
        global_fused = torch.sum(attn_weights * fused, dim=0, keepdim=True)
        whole_logits = self.whole_classifier(global_fused)
        pred_mask = self.seg_head(global_fused).view(1, 1, 64, 64)
        return whole_logits, patch_logits, pred_mask, attn_weights.squeeze(-1)

class ProductionForensicPipeline:
    def __init__(self):
        # Load Backbone
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.extractor = backbone.features.to(DEVICE).eval()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(DEVICE)
        for p in self.extractor.parameters(): p.requires_grad = False
        
        # Load V5-CAG Spatial Specialist
        self.v5_model = V5CAGModel().to(DEVICE).eval()
        if os.path.exists(V5_CHECKPOINT):
            self.v5_model.load_state_dict(torch.load(V5_CHECKPOINT, map_location=DEVICE))
            
        self.transform_norm = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def analyze(self, image_path: str, save_heatmap: bool = True) -> Dict[str, Any]:
        """Runs autonomous end-to-end multi-specialist forensic analysis."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        
        # 1. Global Context Embedding
        g_tensor = self.transform_norm(img).unsqueeze(0).to(DEVICE)
        g_feat = self.pool(self.extractor(g_tensor)).flatten(1)
        
        # 2. Multi-Scale Hierarchical Crops
        patch_scales = [512, 768, 1024]
        p_tensors, p_coords, raw_boxes = [], [], []
        
        for scale in patch_scales:
            step = int(scale * 0.75)
            for y in range(0, max(1, h - scale + 1), max(1, step)):
                for x in range(0, max(1, w - scale + 1), max(1, step)):
                    p_img = img.crop((x, y, x + scale, y + scale))
                    p_tensors.append(self.transform_norm(p_img))
                    p_coords.append([x / w, y / h, scale / w, scale / h, scale / 1024.0])
                    raw_boxes.append((x, y, scale, scale))
                    if len(p_tensors) >= 16: break
                if len(p_tensors) >= 16: break
                
        if len(p_tensors) == 0:
            p_tensors.append(self.transform_norm(img))
            p_coords.append([0.0, 0.0, 1.0, 1.0, 1.0])
            raw_boxes.append((0, 0, w, h))

        # Chunked Feature Extraction
        p_feat_list = []
        for i in range(0, len(p_tensors), 16):
            chunk = torch.stack(p_tensors[i:i+16]).to(DEVICE)
            p_feat_list.append(self.pool(self.extractor(chunk)).flatten(1))
        p_feats = torch.cat(p_feat_list, dim=0)
        p_coords_t = torch.tensor(p_coords, dtype=torch.float32, device=DEVICE)
        
        # 3. Spatial Forward Pass
        whole_logits, patch_logits, pred_mask, attn_weights = self.v5_model(g_feat, p_feats, p_coords_t)
        
        class_probs = F.softmax(whole_logits, dim=-1)[0].cpu().numpy()
        patch_probs = torch.sigmoid(patch_logits).cpu().numpy()
        attn_np = attn_weights.cpu().numpy()
        
        p_real = float(class_probs[0])
        p_partial = float(class_probs[1])
        p_full = float(class_probs[2])
        
        max_patch_prob = float(np.max(patch_probs))
        mean_patch_prob = float(np.mean(patch_probs))
        
        # 4. Heatmap & Suspicious Regions Reconstruction
        full_heatmap = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        suspicious_regions = []
        
        for i, (bx, by, bw, bh) in enumerate(raw_boxes):
            prob = float(patch_probs[i])
            full_heatmap[by:by+bh, bx:bx+bw] += prob
            weight_accum[by:by+bh, bx:bx+bw] += 1.0
            if prob > 0.50:
                suspicious_regions.append({
                    "bbox": [bx, by, bw, bh],
                    "probability": round(prob, 4),
                    "scale": bw
                })
                
        weight_accum[weight_accum == 0] = 1.0
        normalized_heatmap = full_heatmap / weight_accum
        
        # 5. Deterministic Production Decision Logic
        # Condition 1: Pure Real (Zero Patch Anomaly + Low Global Full Score)
        if max_patch_prob < 0.20 and p_full < 0.20:
            verdict = "REAL"
            confidence = max(p_real, 1.0 - max_patch_prob)
            affected_area = 0.0
            suspicious_regions = []
        # Condition 2: Localized Inpainting / Partial AI (High Local Patch Anomaly)
        elif max_patch_prob >= 0.50 and (p_partial > 0.25 or len(suspicious_regions) > 0) and p_full < 0.70:
            verdict = "PARTIAL_AIGC"
            confidence = max(p_partial, max_patch_prob)
            mask_64 = (pred_mask[0, 0].cpu().numpy() > 0.40).astype(np.float32)
            affected_area = float(np.mean(mask_64) * 100.0)
            if affected_area == 0.0 and len(suspicious_regions) > 0:
                # Estimate from bounding box union
                affected_area = min(100.0, len(suspicious_regions) * 12.5)
        # Condition 3: Full AI Synthesis
        elif p_full >= 0.50 or mean_patch_prob >= 0.55:
            verdict = "FULL_AIGC"
            confidence = max(p_full, mean_patch_prob)
            affected_area = 100.0
        # Condition 4: Fallback based on dominant probability
        else:
            if p_real >= max(p_partial, p_full):
                verdict = "REAL"
                confidence = p_real
                affected_area = 0.0
            elif p_partial >= p_full:
                verdict = "PARTIAL_AIGC"
                confidence = p_partial
                affected_area = float(np.mean(pred_mask[0, 0].cpu().numpy() > 0.40) * 100.0)
            else:
                verdict = "FULL_AIGC"
                confidence = p_full
                affected_area = 100.0

        # Save Visual Overlay Heatmap
        heatmap_path = None
        if save_heatmap:
            heatmap_dir = "/home/manan/aigc_robust_detection/reports/production_heatmaps"
            os.makedirs(heatmap_dir, exist_ok=True)
            stem = Path(image_path).stem
            heatmap_path = os.path.join(heatmap_dir, f"{stem}_heatmap.jpg")
            
            heatmap_8u = np.uint8(255 * np.clip(normalized_heatmap, 0, 1))
            color_map = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            if verdict == "REAL":
                # Clean unhighlighted overlay
                overlay = img_bgr
            else:
                overlay = cv2.addWeighted(img_bgr, 0.65, color_map, 0.35, 0)
                for reg in suspicious_regions:
                    rx, ry, rw, rh = reg["bbox"]
                    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
                    
            cv2.imwrite(heatmap_path, overlay)

        return {
            "verdict": verdict,
            "confidence": round(float(confidence), 4),
            "ai_probability": round(float(1.0 - p_real), 4),
            "class_probabilities": {
                "REAL": round(p_real, 4),
                "PARTIAL_AIGC": round(p_partial, 4),
                "FULL_AIGC": round(p_full, 4)
            },
            "affected_area_percentage": round(float(affected_area), 2),
            "max_patch_anomaly": round(max_patch_prob, 4),
            "suspicious_regions_count": len(suspicious_regions),
            "suspicious_regions": suspicious_regions,
            "heatmap_path": heatmap_path
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pipeline = ProductionForensicPipeline()
        res = pipeline.analyze(sys.argv[1])
        print(json.dumps(res, indent=2))
