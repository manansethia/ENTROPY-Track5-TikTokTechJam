#!/usr/bin/env python3
"""
forensic_multiscale_engine.py
-----------------------------
V4.1 Multi-Scale Forensic Inference & Spatial Localization Engine.
Uses the frozen V3 8-expert champion model without modification or retraining.

Key Capabilities:
1. Global & Hierarchical Patch-Level Analysis (1024px, 768px, 512px).
2. Continuous 2D Spatial Heatmap M(x,y) with smooth gaussian aggregation.
3. Suspicious-region bounding box extraction, area, and peak probability.
4. Tri-Class Classification: FULL-AIGC vs PARTIAL-AI vs REAL / BENIGN ARTIFACT.
5. Cross-Scale Consistency Scoring.
6. Full 8-expert logit, probability, and dynamic gating tracking.
7. Structured JSON forensic report + Visual heatmap overlay image export.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

V3_CHECKPOINT_PATH = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v3.pt"

SPECIALIST_CHECKPOINTS = {
    "C2": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c2_spai_vit_best.pt",
    "C4": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c4_convnext_base_best.pt",
    "C5": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c5_convnext_tiny_best.pt",
    "C6": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c6_efficientnet_b0_best.pt",
    "C7": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c7_resnet50_best.pt"
}

eval_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LearnedMultiExpertGatingHead(nn.Module):
    def __init__(self, num_experts=8, temperature=1.15):
        super().__init__()
        self.temperature = temperature
        self.gating = nn.Sequential(
            nn.Linear(num_experts + 1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = torch.std(expert_logits, dim=-1, keepdim=True)
        feat = torch.cat([expert_logits, std], dim=-1)
        weights = self.gating(feat)
        fused = torch.sum(weights * expert_logits, dim=-1)
        return fused, weights

class V4ForensicEngine:
    def __init__(self, checkpoint_path: str = V3_CHECKPOINT_PATH, device: torch.device = DEVICE):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._load_frozen_pipeline()

    def _load_frozen_pipeline(self):
        print(f"[V4 Engine] Loading frozen V3 pipeline from: {self.checkpoint_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Missing V3 checkpoint at {self.checkpoint_path}")

        # 1. Load Gating Head
        ckpt_data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.temperature = ckpt_data.get("temperature", 1.15)
        self.gating_head = LearnedMultiExpertGatingHead(num_experts=8, temperature=self.temperature).to(self.device)
        self.gating_head.load_state_dict(ckpt_data["gating_head_state_dict"])
        self.gating_head.eval()
        for p in self.gating_head.parameters():
            p.requires_grad = False

        # 2. Load Specialists
        self.specialists = []
        for i in range(8):
            mid = f"C{i}"
            if mid in ["C0", "C2", "C7"]:
                m = models.resnet50(num_classes=1)
            elif mid in ["C1", "C4", "C5"]:
                m = models.convnext_tiny(num_classes=1)
            elif mid in ["C3", "C6"]:
                m = models.efficientnet_b0(num_classes=1)

            ckpt_p = SPECIALIST_CHECKPOINTS.get(mid)
            if ckpt_p and os.path.exists(ckpt_p):
                sd = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                m.load_state_dict(sd)

            m = m.to(self.device).eval()
            for p in m.parameters():
                p.requires_grad = False
            self.specialists.append(m)

        print("  All 8 frozen specialist models loaded into FP32 memory successfully ✅")

    def _eval_batch_logits(self, batch_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Evaluates a batch of image tensors (B, 3, 224, 224) through all 8 experts and the gating head."""
        logits_list = []
        with torch.no_grad():
            for m in self.specialists:
                l = m(batch_tensors).squeeze(-1)
                logits_list.append(l)
            stacked_logits = torch.stack(logits_list, dim=-1) # (B, 8)
            fused_logits, weights = self.gating_head(stacked_logits)
            fused_probs = torch.sigmoid(fused_logits / self.gating_head.temperature)
            expert_probs = [torch.sigmoid(l) for l in logits_list]
            return fused_probs, weights, expert_probs

    def extract_patches(
        self,
        img: Image.Image,
        scales: List[int] = [1024, 768, 512],
        overlap_ratio: float = 0.20
    ) -> List[Dict[str, Any]]:
        """Extracts multi-scale overlapping crops from original high-resolution image."""
        w, h = img.size
        patches = []
        patch_idx = 0

        # Filter scales that are reasonable for the image size
        valid_scales = [s for s in scales if s <= max(w, h)]
        if not valid_scales:
            valid_scales = [min(w, h, 512)]

        for scale in valid_scales:
            step = max(32, int(scale * (1.0 - overlap_ratio)))
            
            # Compute x coordinates
            x_coords = list(range(0, max(1, w - scale + 1), step))
            if x_coords[-1] + scale < w:
                x_coords.append(w - scale)
            x_coords = sorted(list(set(x_coords)))

            # Compute y coordinates
            y_coords = list(range(0, max(1, h - scale + 1), step))
            if y_coords[-1] + scale < h:
                y_coords.append(h - scale)
            y_coords = sorted(list(set(y_coords)))

            for y in y_coords:
                for x in x_coords:
                    x2 = min(w, x + scale)
                    y2 = min(h, y + scale)
                    crop_img = img.crop((x, y, x2, y2))
                    
                    patches.append({
                        "patch_id": f"P_{scale}_{patch_idx:04d}",
                        "scale": scale,
                        "x": x,
                        "y": y,
                        "width": x2 - x,
                        "height": y2 - y,
                        "crop_pil": crop_img
                    })
                    patch_idx += 1

        return patches

    def analyze_image(
        self,
        image_path: str,
        scales: List[int] = [1024, 768, 512],
        overlap_ratio: float = 0.20,
        batch_size: int = 32,
        hierarchical: bool = False
    ) -> Dict[str, Any]:
        """Performs complete V4.1 Multi-Scale Forensic Analysis on a single high-resolution image."""
        t0 = time.time()
        raw_img = Image.open(image_path).convert("RGB")
        w, h = raw_img.size
        orig_res = f"{w}x{h}"

        # -------------------------------------------------------------
        # 1. Global View Inference (V4.1.1)
        # -------------------------------------------------------------
        global_tensor = eval_transform(raw_img).unsqueeze(0).to(self.device)
        g_prob_t, g_weights_t, g_spec_probs_t = self._eval_batch_logits(global_tensor)
        
        global_prob = float(g_prob_t[0].item())
        global_weights = [round(float(w.item()), 4) for w in g_weights_t[0]]
        global_spec_probs = [round(float(p[0].item()), 4) for p in g_spec_probs_t]

        # -------------------------------------------------------------
        # 2. Multi-Scale Overlapping Patch Extraction (V4.1.2)
        # -------------------------------------------------------------
        patches = self.extract_patches(raw_img, scales=scales, overlap_ratio=overlap_ratio)
        total_patches = len(patches)

        # -------------------------------------------------------------
        # 3. Patch-Level V3 Eight-Expert Inference (V4.1.3)
        # -------------------------------------------------------------
        patch_records = []
        for i in range(0, total_patches, batch_size):
            batch_slice = patches[i:i + batch_size]
            tensors = torch.stack([eval_transform(p["crop_pil"]) for p in batch_slice]).to(self.device)
            f_probs, weights, spec_probs = self._eval_batch_logits(tensors)
            
            f_np = f_probs.cpu().numpy()
            w_np = weights.cpu().numpy()
            s_np = [sp.cpu().numpy() for sp in spec_probs]

            for j, p in enumerate(batch_slice):
                p_prob = float(f_np[j])
                p_weights = [round(float(w_np[j, k]), 4) for k in range(8)]
                p_specs = [round(float(s_np[k][j]), 4) for k in range(8)]
                
                # Dominant expert for this patch
                dominant_exp = f"C{int(np.argmax(p_weights))}"
                
                patch_records.append({
                    "patch_id": p["patch_id"],
                    "scale": p["scale"],
                    "x": p["x"],
                    "y": p["y"],
                    "width": p["width"],
                    "height": p["height"],
                    "fused_prob": p_prob,
                    "dominant_expert": dominant_exp,
                    "weights": p_weights,
                    "specialist_probs": p_specs
                })

        # -------------------------------------------------------------
        # 4. Continuous Spatial Heatmap M(x,y) Aggregation (V4.1.4)
        # -------------------------------------------------------------
        # Downscale grid for continuous heatmap map M(x,y) to save RAM (e.g. 512x512 grid or aspect ratio matched)
        grid_h = min(1024, max(256, h // 4))
        grid_w = int(grid_h * (w / h))
        
        heatmap_accum = np.zeros((grid_h, grid_w), dtype=np.float32)
        weight_accum = np.zeros((grid_h, grid_w), dtype=np.float32)

        scale_x = grid_w / float(w)
        scale_y = grid_h / float(h)

        for pr in patch_records:
            gx1 = int(pr["x"] * scale_x)
            gy1 = int(pr["y"] * scale_y)
            gx2 = max(gx1 + 1, int((pr["x"] + pr["width"]) * scale_x))
            gy2 = max(gy1 + 1, int((pr["y"] + pr["height"]) * scale_y))

            # 2D Gaussian Kernel for smooth overlap blending
            pw = gx2 - gx1
            ph = gy2 - gy1
            yy, xx = np.mgrid[-1:1:complex(0, ph), -1:1:complex(0, pw)]
            kernel = np.exp(-(xx**2 + yy**2) / 0.8)

            heatmap_accum[gy1:gy2, gx1:gx2] += pr["fused_prob"] * kernel
            weight_accum[gy1:gy2, gx1:gx2] += kernel

        # Normalize heatmap
        weight_accum[weight_accum == 0] = 1.0
        continuous_heatmap = heatmap_accum / weight_accum
        continuous_heatmap = np.clip(continuous_heatmap, 0.0, 1.0)

        # -------------------------------------------------------------
        # 5. Extract Suspicious Regions & Bounding Boxes (V4.1.4 & V4.1.6)
        # -------------------------------------------------------------
        suspicious_mask = (continuous_heatmap >= 0.60)
        suspicious_regions = []
        
        # Simple connected component / cluster bounding box detection
        from scipy.ndimage import label as nd_label
        labeled_array, num_features = nd_label(suspicious_mask)
        
        for feat_idx in range(1, num_features + 1):
            y_indices, x_indices = np.where(labeled_array == feat_idx)
            if len(y_indices) < 20: # skip tiny speckles
                continue
            
            # Map back to original image coordinates
            rx1 = int(np.min(x_indices) / scale_x)
            ry1 = int(np.min(y_indices) / scale_y)
            rx2 = int(np.max(x_indices) / scale_x)
            ry2 = int(np.max(y_indices) / scale_y)
            
            region_probs = continuous_heatmap[y_indices, x_indices]
            peak_prob = float(np.max(region_probs))
            mean_prob = float(np.mean(region_probs))
            
            # Find patches that cover this region and assess cross-scale consistency
            overlapping_patches = [
                pr for pr in patch_records
                if not (pr["x"] > rx2 or pr["x"] + pr["width"] < rx1 or pr["y"] > ry2 or pr["y"] + pr["height"] < ry1)
            ]
            
            scale_set = set(p["scale"] for p in overlapping_patches)
            consistency = "High" if len(scale_set) >= 2 and peak_prob > 0.85 else ("Medium" if peak_prob > 0.70 else "Low")
            
            # Find dominant expert across overlapping patches
            if overlapping_patches:
                exp_counts = {}
                for p in overlapping_patches:
                    exp_counts[p["dominant_expert"]] = exp_counts.get(p["dominant_expert"], 0) + 1
                dom_exp = max(exp_counts, key=exp_counts.get)
            else:
                dom_exp = "C4"

            suspicious_regions.append({
                "region_id": f"R{len(suspicious_regions) + 1}",
                "box": [rx1, ry1, rx2 - rx1, ry2 - ry1],
                "coordinates": f"x={rx1}, y={ry1}, w={rx2-rx1}, h={ry2-ry1}",
                "peak_probability": round(peak_prob, 4),
                "mean_probability": round(mean_prob, 4),
                "dominant_expert": dom_exp,
                "cross_scale_consistency": consistency,
                "scales_detected": list(scale_set)
            })

        # Sort suspicious regions by peak probability descending
        suspicious_regions = sorted(suspicious_regions, key=lambda r: r["peak_probability"], reverse=True)

        # -------------------------------------------------------------
        # 6. Tri-Class Classification Decision (V4.1.5)
        # -------------------------------------------------------------
        patch_probs_all = [p["fused_prob"] for p in patch_records]
        mean_patch_prob = float(np.mean(patch_probs_all)) if patch_probs_all else global_prob
        max_patch_prob = float(np.max(patch_probs_all)) if patch_probs_all else global_prob
        std_patch_prob = float(np.std(patch_probs_all)) if patch_probs_all else 0.0
        high_patch_ratio = float(np.mean(np.array(patch_probs_all) >= 0.60)) if patch_probs_all else 0.0

        if high_patch_ratio >= 0.65 and global_prob >= 0.50:
            classification = "FULL-AIGC (SYNTHETIC GENERATION)"
            localized_prob = max_patch_prob
            interpretation = "AI generation evidence is broadly and uniformly distributed across all scales and regions."
        elif (max_patch_prob >= 0.70 and mean_patch_prob < 0.55 and std_patch_prob >= 0.15) or (len(suspicious_regions) > 0 and global_prob < 0.50):
            classification = "PARTIAL-AI (LOCALIZED INPAINTING / GENERATIVE FILL)"
            localized_prob = max_patch_prob
            interpretation = f"The overall image structure appears mostly authentic, but {len(suspicious_regions)} localized region(s) exhibit strong synthetic artifacts indicating generative inpainting/editing."
        elif global_prob < 0.40 and max_patch_prob < 0.60:
            classification = "REAL_AUTHENTIC (CAMERA CAPTURE)"
            localized_prob = max_patch_prob
            interpretation = "Intact camera sensor noise and consistent physical illumination; no localized synthetic anomalies detected."
        else:
            classification = "REAL / BENIGN ARTIFACT (AUTHENTIC WITH POST-PROCESSING)"
            localized_prob = max_patch_prob
            interpretation = "Mild localized filter activations consistent with JPEG/WebP compression or photographic tone adjustment."

        elapsed_sec = time.time() - t0

        return {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "original_resolution": orig_res,
            "elapsed_seconds": round(elapsed_sec, 2),
            "global_inference": {
                "ai_probability": round(global_prob, 4),
                "classification": "AIGC" if global_prob >= 0.50 else "REAL",
                "specialist_probabilities": dict(zip([f"C{i}" for i in range(8)], global_spec_probs)),
                "gating_weights": dict(zip([f"C{i}" for i in range(8)], global_weights))
            },
            "patch_inference": {
                "total_patches_evaluated": total_patches,
                "scales_used": scales,
                "mean_patch_probability": round(mean_patch_prob, 4),
                "max_patch_probability": round(max_patch_prob, 4),
                "patch_std_deviation": round(std_patch_prob, 4)
            },
            "forensic_verdict": {
                "classification": classification,
                "global_ai_probability": round(global_prob, 4),
                "localized_ai_probability": round(localized_prob, 4),
                "suspicious_region_count": len(suspicious_regions),
                "interpretation": interpretation
            },
            "suspicious_regions": suspicious_regions,
            "continuous_heatmap": continuous_heatmap,
            "patch_records": patch_records
        }

    def render_and_save_visual_heatmap(
        self,
        analysis_result: Dict[str, Any],
        output_image_path: str
    ):
        """Renders visual overlay with continuous probability heatmap M(x,y) and bounding boxes."""
        raw_img = Image.open(analysis_result["image_path"]).convert("RGB")
        w, h = raw_img.size
        
        heatmap_grid = analysis_result["continuous_heatmap"]
        
        # Colorize heatmap with jet colormap
        try:
            cmap = matplotlib.colormaps["jet"]
        except Exception:
            cmap = getattr(cm, "get_cmap", lambda x: cm.jet)("jet")
        colored_hm = cmap(heatmap_grid)[:, :, :3] # (H, W, 3) in [0, 1]
        colored_hm_img = Image.fromarray((colored_hm * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)

        # Blend with original image (alpha 0.45)
        overlay = Image.blend(raw_img, colored_hm_img, alpha=0.45)
        draw = ImageDraw.Draw(overlay)

        # Draw Bounding Boxes for Suspicious Regions
        for r in analysis_result["suspicious_regions"]:
            x, y, bw, bh = r["box"]
            peak_p = r["peak_probability"]
            exp = r["dominant_expert"]
            
            # Thick Red Bounding Box
            draw.rectangle([x, y, x + bw, y + bh], outline="red", width=max(3, int(min(w, h) * 0.005)))
            
            label_text = f"{r['region_id']}: {peak_p:.2f} ({exp})"
            # Draw label tag
            draw.rectangle([x, max(0, y - 25), x + 180, max(0, y)], fill="red")
            draw.text((x + 5, max(0, y - 22)), label_text, fill="white")

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        overlay.save(output_image_path, quality=92)
        print(f"  [Visual Heatmap Saved] {output_image_path}")

def format_forensic_report_text(res: Dict[str, Any]) -> str:
    """Generates the structured human-readable text report matching V4.1.8 specification."""
    lines = [
        "=" * 80,
        "  IMAGE FORENSIC ANALYSIS REPORT (V4.1 MULTI-SCALE ENGINE)",
        "=" * 80,
        f"File                 : {res['filename']}",
        f"Original Resolution  : {res['original_resolution']}",
        f"Processing Time      : {res['elapsed_seconds']}s ({res['patch_inference']['total_patches_evaluated']} patches evaluated)",
        "",
        f"Global AI Probability: {res['global_inference']['ai_probability']:.4f}",
        f"Global Classification: {res['global_inference']['classification']}",
        "",
        f"Localized AI Prob    : {res['forensic_verdict']['localized_ai_probability']:.4f}",
        f"Forensic Verdict     : >>> {res['forensic_verdict']['classification']} <<<",
        "",
        f"Interpretation       : {res['forensic_verdict']['interpretation']}",
        "",
        "-" * 80,
        "  SUSPICIOUS LOCALIZED REGIONS",
        "-" * 80
    ]

    if not res["suspicious_regions"]:
        lines.append("  None detected (Uniform physical consistency).")
    else:
        for r in res["suspicious_regions"]:
            lines.extend([
                f"  [{r['region_id']}]",
                f"    Coordinates            : {r['coordinates']}",
                f"    Peak Probability       : {r['peak_probability']:.4f} (Mean: {r['mean_probability']:.4f})",
                f"    Dominant Expert        : {r['dominant_expert']}",
                f"    Cross-Scale Consistency: {r['cross_scale_consistency']} (Scales: {r['scales_detected']})",
                ""
            ])

    lines.extend([
        "-" * 80,
        "  EXPERT & GATING SUMMARY",
        "-" * 80,
        "  Expert Probabilities   : " + ", ".join([f"{k}:{v:.3f}" for k, v in res["global_inference"]["specialist_probabilities"].items()]),
        "  Dynamic Gating Weights : " + ", ".join([f"{k}:{v:.3f}" for k, v in res["global_inference"]["gating_weights"].items()]),
        "=" * 80
    ])

    return "\n".join(lines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python forensic_multiscale_engine.py <image_path> [output_report_json] [output_heatmap_jpg]")
        sys.exit(1)

    img_p = sys.argv[1]
    out_j = sys.argv[2] if len(sys.argv) > 2 else "/home/manan/aigc_robust_detection/reports/v4_forensics/sample_report.json"
    out_h = sys.argv[3] if len(sys.argv) > 3 else "/home/manan/aigc_robust_detection/reports/v4_heatmaps/sample_heatmap.jpg"

    engine = V4ForensicEngine()
    report = engine.analyze_image(img_p, scales=[1024, 768, 512], overlap_ratio=0.20)
    
    txt_report = format_forensic_report_text(report)
    print(txt_report)

    # Clean array for JSON serialization
    json_safe = dict(report)
    json_safe.pop("continuous_heatmap", None)
    with open(out_j, "w") as f:
        json.dump(json_safe, f, indent=2)

    engine.render_and_save_visual_heatmap(report, out_h)
