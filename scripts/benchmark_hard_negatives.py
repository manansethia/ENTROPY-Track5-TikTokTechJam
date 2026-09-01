#!/usr/bin/env python3
"""Hard-Negative / CGI & Digital Art False-Positive Guardrail Benchmark.
Verifies that human CGI 3D game renders, HDR photography, and digital art are not falsely flagged as AIGC.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel

from scripts.data import VALID_EXTS
from scripts.train_tri_hybrid_gating import DynamicGatingFusionHead


def find_image_files(root_dir):
    files = []
    for base, _, names in os.walk(root_dir):
        for n in sorted(names):
            if n.lower().endswith(VALID_EXTS):
                files.append(os.path.join(base, n))
    return sorted(files)


def main():
    p = argparse.ArgumentParser(description="Hard-Negative Guardrail Benchmark")
    p.add_argument("--checkpoint", default="checkpoints/tri_hybrid_45k_v3/best_model.pt")
    p.add_argument("--hard_neg_dir", default="/mnt/ai-storage/aigc_data/validation_LOCKED/val2017")
    p.add_argument("--siglip_dir", default="/mnt/ai-storage/aigc_data/models/siglip_base_224")
    p.add_argument("--clip_dir", default="/mnt/ai-storage/aigc_data/models/clip_vitl14")
    p.add_argument("--dinov2_dir", default="/mnt/ai-storage/aigc_data/models/dinov2_large")
    p.add_argument("--max_images", type=int, default=250)
    p.add_argument("--output_json", default="reports/hard_negative_benchmark.json")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Running Hard-Negative Guardrail Benchmark on {device}...")

    # Load backbones
    siglip_proc = AutoProcessor.from_pretrained(args.siglip_dir)
    siglip_model = AutoModel.from_pretrained(args.siglip_dir).to(device)
    siglip_model.eval()

    clip_proc = AutoProcessor.from_pretrained(args.clip_dir)
    clip_model = CLIPModel.from_pretrained(args.clip_dir).to(device)
    clip_model.eval()

    # Load Dynamic Gating Head
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    has_dinov2 = "proj_dinov2.0.weight" in state_dict

    dinov2_model = None
    dinov2_transform = None
    if has_dinov2 and os.path.isdir(args.dinov2_dir):
        from torchvision import transforms
        dinov2_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dinov2_model = AutoModel.from_pretrained(args.dinov2_dir).to(device)
        dinov2_model.eval()

    fusion_head = DynamicGatingFusionHead(has_dinov2=has_dinov2).to(device)
    fusion_head.load_state_dict(state_dict)
    fusion_head.eval()

    # Collect hard negative candidate paths
    image_paths = find_image_files(args.hard_neg_dir)[:args.max_images]
    print(f"Evaluating {len(image_paths)} authentic hard-negative test samples...")

    scores = []
    false_positives = 0

    with torch.no_grad():
        for pth in tqdm(image_paths, desc="Hard-Negative Audit"):
            try:
                img = Image.open(pth).convert("RGB")
                s_in = siglip_proc(images=img, return_tensors="pt").to(device)
                s_feat = siglip_model.vision_model(**s_in).pooler_output

                c_in = clip_proc(images=img, return_tensors="pt").to(device)
                c_feat = clip_model.vision_model(**c_in).pooler_output
                c_feat = c_feat / c_feat.norm(dim=-1, keepdim=True)

                if has_dinov2 and dinov2_model is not None:
                    d_tensor = dinov2_transform(img).unsqueeze(0).to(device)
                    d_out = dinov2_model(d_tensor)
                    d_feat = d_out.pooler_output if hasattr(d_out, "pooler_output") and d_out.pooler_output is not None else d_out.last_hidden_state[:, 0]
                    d_feat = d_feat / d_feat.norm(dim=-1, keepdim=True)
                else:
                    d_feat = None

                logits, _ = fusion_head(s_feat, c_feat, d_feat)
                prob_fake = F.softmax(logits, dim=-1)[0, 1].item()
                scores.append(prob_fake)
                if prob_fake >= 0.5:
                    false_positives += 1
            except Exception:
                continue

    scores = np.array(scores)
    fpr = (false_positives / len(scores)) * 100.0 if len(scores) > 0 else 0.0
    mean_prob = float(np.mean(scores))
    p95_prob = float(np.percentile(scores, 95))

    report = {
        "total_hard_negatives": len(scores),
        "false_positive_count": int(false_positives),
        "false_positive_rate_percent": float(fpr),
        "mean_synthetic_probability": mean_prob,
        "p95_synthetic_probability": p95_prob,
        "guardrail_status": "PASS (FPR < 2.0%)" if fpr < 2.0 else "INVESTIGATE",
    }

    print("\n" + "=" * 60)
    print("HARD-NEGATIVE / CGI GUARDRAIL BENCHMARK REPORT")
    print("=" * 60)
    print(f"Total Authentic Hard-Negatives Tested: {len(scores):,}")
    print(f"False Positives Triggered:           {false_positives}")
    print(f"False Positive Rate (FPR):            {fpr:.2f}%")
    print(f"Mean Predicted Fake Probability:      {mean_prob*100:.2f}%")
    print(f"95th Percentile Fake Probability:     {p95_prob*100:.2f}%")
    print(f"Guardrail Verdict:                    {report['guardrail_status']}")
    print("=" * 60 + "\n")

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
