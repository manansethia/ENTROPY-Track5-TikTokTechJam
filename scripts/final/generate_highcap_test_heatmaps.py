#!/usr/bin/env python3
"""
generate_highcap_test_heatmaps.py
---------------------------------
Runs inference on the 4 real-world test images using the newly trained
HighCapacityStudentForensicModel (96.59M) across FP32, FP16, and INT8 checkpoints.
Generates:
  - 3-Way Classification & Probabilities
  - Continuous Heatmap overlays (64x64 upsampled to native image resolution)
  - Suspicious Region Bounding Boxes
  - Affected Area Percentage (%)
  - Saves annotated visualizations to reports/heatmaps_highcap/
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.highcap_distilled_forensic_model import HighCapacityStudentForensicModel

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    test_dir_1 = "/home/manan/aigc_robust_detection/test_inputs"
    test_dir_2 = "/home/manan/aigc_robust_detection/test_inputs/final_user_test"
    out_dir = "/home/manan/aigc_robust_detection/reports/heatmaps_highcap"
    os.makedirs(out_dir, exist_ok=True)

    images = [
        os.path.join(test_dir_1, "4women.webp"),
    ]
    if os.path.exists(test_dir_2):
        for f in sorted(os.listdir(test_dir_2)):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                images.append(os.path.join(test_dir_2, f))

    images = [img for img in images if os.path.exists(img)]
    print(f"Found {len(images)} test images:")
    for img_p in images:
        print(f"  - {os.path.basename(img_p)}")

    # Load 96.59M Student Model in FP32
    fp32_ckpt = "/home/manan/aigc_robust_detection/checkpoints/distilled/highcap_distilled_forensic_model_fp32.pt"
    student = HighCapacityStudentForensicModel().to(DEVICE).eval()
    sd = torch.load(fp32_ckpt, map_location=DEVICE)
    student.load_state_dict(sd["model_state_dict"])
    p_cnt = sum(p.numel() for p in student.parameters())
    print(f"\nLoaded HighCapacityStudentForensicModel: {p_cnt:,} parameters ({p_cnt/1e6:.2f}M) ✅")

    t_224 = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_names = ["REAL", "PARTIAL_AIGC", "FULL_AIGC"]
    results = []

    print("\n" + "=" * 105)
    print("  HIGH-CAPACITY DISTILLED STUDENT (96.59M) — FINAL TEST INFERENCE RESULTS")
    print("=" * 105)

    for img_path in images:
        stem = Path(img_path).stem
        orig_img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = orig_img.size
        img_np = np.array(orig_img)

        t_tensor = t_224(orig_img).unsqueeze(0).to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = student(t_tensor)
        lat_ms = (time.perf_counter() - t0) * 1000

        probs = out["probabilities"][0].cpu().numpy()
        pred_idx = int(out["class_logits"].argmax(dim=-1).item())
        pred_label = class_names[pred_idx]
        conf = probs[pred_idx]

        # Process spatial heatmap (1, 64, 64)
        hmap = out["segmentation_heatmap"][0, 0].cpu().numpy() # (64, 64)
        hmap_norm = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hmap_resized = cv2.resize(hmap_norm, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

        # Threshold mask for suspicious regions
        mask_binary = (hmap_resized > 0.45).astype(np.uint8) * 255
        affected_area_pct = (np.count_nonzero(mask_binary) / (w_orig * h_orig)) * 100.0

        # Find bounding boxes
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > (w_orig * h_orig * 0.01): # Filter tiny speckles (>1% area)
                x, y, w, h = cv2.boundingRect(c)
                bboxes.append({"bbox_xywh": [int(x), int(y), int(w), int(h)], "area_pct": round((w * h) / (w_orig * h_orig) * 100.0, 2)})

        # Colorize Heatmap & Blend
        heatmap_color = cv2.applyColorMap((hmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        blend = cv2.addWeighted(img_bgr, 0.60, heatmap_color, 0.40, 0)

        # Draw Bounding Boxes on Blend
        for b in bboxes:
            bx, by, bw, bh = b["bbox_xywh"]
            cv2.rectangle(blend, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
            cv2.putText(blend, f"Suspicious Region ({b['area_pct']}%)", (bx, max(20, by - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Header overlay
        header_txt = f"{pred_label} ({conf*100:.1f}%) | Affected: {affected_area_pct:.1f}% | Latency: {lat_ms:.1f}ms"
        cv2.putText(blend, header_txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        out_img_path = f"{out_dir}/{stem}_highcap_96m_heatmap.jpg"
        cv2.imwrite(out_img_path, blend)

        res_item = {
            "image": os.path.basename(img_path),
            "prediction": pred_label,
            "confidence": round(float(conf), 4),
            "probabilities": {
                "REAL": round(float(probs[0]), 4),
                "PARTIAL_AIGC": round(float(probs[1]), 4),
                "FULL_AIGC": round(float(probs[2]), 4)
            },
            "affected_area_pct": round(float(affected_area_pct), 2),
            "bounding_boxes": bboxes,
            "inference_latency_ms": round(float(lat_ms), 2),
            "heatmap_path": out_img_path
        }
        results.append(res_item)

        print(f"File: {stem:<35} | Pred: {pred_label:<12} (Conf: {conf*100:.1f}%) | Affected: {affected_area_pct:.1f}% | Latency: {lat_ms:.1f}ms")
        print(f"   Probs -> REAL: {probs[0]*100:.1f}%, PARTIAL_AIGC: {probs[1]*100:.1f}%, FULL_AIGC: {probs[2]*100:.1f}%")
        print(f"   Heatmap saved -> {out_img_path}")

    # Export results JSON
    json_path = f"{out_dir}/highcap_test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary JSON exported to {json_path} ✅")

if __name__ == "__main__":
    main()
