#!/usr/bin/env python3
"""
v5_inference_engine.py
-----------------------
V5 Unified Forensic Inference Engine.
Exposes:
  analyze(image_path) -> {
      "verdict": "REAL" | "PARTIAL_AIGC" | "FULL_AIGC",
      "confidence": float,
      "ai_probability": float,
      "affected_area_percentage": float,
      "localization_mask": str (path to saved heatmap PNG),
      "suspicious_regions": [
          {"bbox": [x, y, w, h], "probability": float, "scale": int, "evidence_type": str}
      ],
      "patch_evidence": [...],
      "provenance": {...},
      "forensic_report": str
  }
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/home/manan/aigc_robust_detection/checkpoints/experimental/v5/v5_champion_cag.pt"
HEATMAP_OUT_DIR = "/home/manan/aigc_robust_detection/reports/v5/heatmaps"
os.makedirs(HEATMAP_OUT_DIR, exist_ok=True)

# Import V5CAGModel architecture
from train_v5_master_cag import V5CAGModel
from v5_provenance_engine import V5ProvenanceEngine

class V5ForensicInferenceEngine:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        print(f"  [Inference Engine] Loading ConvNeXt-Tiny Feature Backbone...")
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.extractor = backbone.features.to(device).eval()
        self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        self.transform_norm = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = V5CAGModel().to(device)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"  [Inference Engine] Loaded V5-CAG Model Checkpoint from {checkpoint_path} ✅")
        else:
            print(f"  [Inference Engine] Checkpoint not yet trained at {checkpoint_path}")
        self.model.eval()
        
        self.provenance_engine = V5ProvenanceEngine()

    @torch.no_grad()
    def analyze(self, image_path: str, save_heatmap: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        if not os.path.exists(image_path):
            return {"error": f"Image not found at {image_path}"}
            
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        
        # 1. Global View Representation
        g_tensor = self.transform_norm(img).unsqueeze(0).to(device)
        g_feat = self.pool(self.extractor(g_tensor)).flatten(1) # (1, 768)
        
        # 2. Hierarchical Multi-Scale Overlapping Patch Extraction
        patch_scales = [512, 768, 1024]
        p_tensors = []
        p_coords = []
        raw_boxes = []
        
        for scale in patch_scales:
            step = int(scale * 0.75)
            for y in range(0, max(1, h - scale + 1), max(1, step)):
                for x in range(0, max(1, w - scale + 1), max(1, step)):
                    p_img = img.crop((x, y, x + scale, y + scale))
                    p_tensor = self.transform_norm(p_img)
                    p_tensors.append(p_tensor)
                    p_coords.append([x / w, y / h, scale / w, scale / h, scale / 1024.0])
                    raw_boxes.append((x, y, scale, scale))
                    
        if len(p_tensors) == 0:
            p_tensors.append(self.transform_norm(img))
            p_coords.append([0.0, 0.0, 1.0, 1.0, 1.0])
            raw_boxes.append((0, 0, w, h))

        # Chunked patch feature extraction to ensure 0 OOM on 8K/12K/100MP images
        p_feat_list = []
        for i in range(0, len(p_tensors), 16):
            chunk = torch.stack(p_tensors[i:i+16]).to(device)
            p_feat_list.append(self.pool(self.extractor(chunk)).flatten(1))
        p_feats = torch.cat(p_feat_list, dim=0) # (N, 768)
        p_coords_tensor = torch.tensor(p_coords, dtype=torch.float32, device=device)
        
        # 3. Model Forward Pass
        whole_logits, patch_logits, pred_mask, attn_weights = self.model(g_feat, p_feats, p_coords_tensor)
        
        # Probabilities & Class Verdict
        class_probs = F.softmax(whole_logits, dim=-1)[0].cpu().numpy()
        patch_probs = torch.sigmoid(patch_logits).cpu().numpy()
        attn_np = attn_weights.cpu().numpy()
        
        verdict_idx = int(np.argmax(class_probs))
        verdict_map = {0: "REAL", 1: "PARTIAL_AIGC", 2: "FULL_AIGC"}
        verdict = verdict_map[verdict_idx]
        confidence = float(class_probs[verdict_idx])
        ai_prob = float(class_probs[1] + class_probs[2])
        
        # 4. Heatmap Reconstruction on Original High-Res Coordinate Grid
        full_heatmap = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        
        suspicious_regions = []
        patch_evidence = []
        
        for i, (bx, by, bw, bh) in enumerate(raw_boxes):
            p_prob = float(patch_probs[i])
            p_attn = float(attn_np[i])
            
            full_heatmap[by:by+bh, bx:bx+bw] += p_prob
            weight_accum[by:by+bh, bx:bx+bw] += 1.0
            
            patch_evidence.append({
                "patch_id": i,
                "bbox": [bx, by, bw, bh],
                "scale": bw,
                "ai_probability": round(p_prob, 4),
                "attention_weight": round(p_attn, 4)
            })
            
            if p_prob > 0.50:
                suspicious_regions.append({
                    "bbox": [bx, by, bw, bh],
                    "probability": round(p_prob, 4),
                    "scale": bw,
                    "evidence_type": "localized_synthetic_frequency_or_diffusion_anomaly"
                })
                
        # Normalize continuous heatmap
        valid_mask = weight_accum > 0
        full_heatmap[valid_mask] /= weight_accum[valid_mask]
        
        # Estimate affected area percentage
        affected_area_pct = float(np.mean(full_heatmap > 0.45) * 100.0)
        
        # Save Heatmap Visualization
        heatmap_path = None
        if save_heatmap:
            heatmap_norm = np.uint8(np.clip(full_heatmap * 255.0, 0, 255))
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_bgr, 0.65, heatmap_color, 0.35, 0)
            
            out_name = f"{Path(image_path).stem}_v5_heatmap.jpg"
            heatmap_path = os.path.join(HEATMAP_OUT_DIR, out_name)
            cv2.imwrite(heatmap_path, overlay)

        # 5. Independent Decoupled Provenance Audit
        provenance = self.provenance_engine.analyze_provenance(image_path)
        
        infer_time = time.time() - t0
        
        forensic_report = f"""
========================================================================================
  AIGC FORENSICS V5 EVIDENCE REPORT
========================================================================================
  Image File: {image_path}
  Resolution: {w} x {h} ({w*h/1e6:.2f} Megapixels)
  Inference Time: {infer_time*1000:.1f} ms

  FORENSIC VERDICT: {verdict} (Confidence: {confidence*100:.2f}%)
    - Real Probability        : {class_probs[0]*100:.2f}%
    - Partial-AIGC Probability: {class_probs[1]*100:.2f}%
    - Full-AIGC Probability   : {class_probs[2]*100:.2f}%
    - Aggregate AI Probability: {ai_prob*100:.2f}%

  SPATIAL LOCALIZATION FINDINGS:
    - Estimated Affected Area : {affected_area_pct:.2f}%
    - Suspicious Regions Count: {len(suspicious_regions)}
    - Heatmap Artifact Saved  : {heatmap_path}

  PROVENANCE & METADATA AUDIT:
    - Provenance Verdict      : {provenance.get('provenance_verdict')}
    - C2PA Content Credentials: {provenance.get('c2pa', {}).get('c2pa_status')}
    - Camera Make / Model     : {provenance.get('camera', {}).get('make')} {provenance.get('camera', {}).get('model')}
    - Software / Editor       : {provenance.get('software')}
========================================================================================
        """.strip()

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "ai_probability": round(ai_prob, 4),
            "class_probabilities": {
                "REAL": round(float(class_probs[0]), 4),
                "PARTIAL_AIGC": round(float(class_probs[1]), 4),
                "FULL_AIGC": round(float(class_probs[2]), 4)
            },
            "affected_area_percentage": round(affected_area_pct, 2),
            "localization_heatmap_path": heatmap_path,
            "suspicious_regions": suspicious_regions,
            "patch_evidence": patch_evidence[:15], # Top 15 patches
            "provenance": provenance,
            "forensic_report": forensic_report,
            "inference_time_ms": round(infer_time * 1000, 1)
        }

if __name__ == "__main__":
    engine = V5ForensicInferenceEngine()
    test_img = "/mnt/ai-storage/aigc_data/datasets/ultra_highres_gigapixel_pool/real_dslr_3k_10k/real_ultra_highres_0016_6016x4016.jpg"
    if os.path.exists(test_img):
        res = engine.analyze(test_img)
        print(res["forensic_report"])
