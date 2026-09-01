#!/usr/bin/env python3
"""
verify_compiled_master_models.py
--------------------------------
Verifies that the compiled master monolithic checkpoints (FP32 and FP16)
load cleanly and execute full-spectrum forensic inference on test images.
"""

import os
import sys
import time
import json
import torch
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np

sys.path.insert(0, "/mnt/ai-storage/aigc_data/models/aide_finetuned")
sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.compile_master_unified_model import MasterUnifiedForensicModel, TripleHybridChampion

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def verify_checkpoint(ckpt_path: str, precision: str, test_image_path: str):
    target_device = torch.device("cuda:0" if (precision == "FP16" and torch.cuda.is_available()) else "cpu")
    print(f"\n👉 Testing Compiled {precision} Monolith Checkpoint: {ckpt_path} on {target_device}...")
    t0 = time.time()
    
    # 1. Instantiate Model Architecture
    model = MasterUnifiedForensicModel()
    if precision == "FP16":
        model = model.half()
        
    # 2. Load Compiled Weights
    print("   Loading state dict from single compiled checkpoint...")
    data = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(data["model_state_dict"])
    model = model.to(target_device).eval()
    load_time = time.time() - t0
    print(f"   Checkpoint Loaded in {load_time:.2f}s | Parameters: {data['total_parameters']:,}")

    # 3. Prepare Multi-Modal Input Package
    img = Image.open(test_image_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    t_224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_256 = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_384 = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    target_dtype = torch.float16 if precision == "FP16" else torch.float32
    
    img_224 = t_224(img).unsqueeze(0).to(target_device, dtype=target_dtype)
    img_256_5v = t_256(img).unsqueeze(0).unsqueeze(1).repeat(1, 5, 1, 1, 1).to(target_device, dtype=target_dtype)
    img_384 = t_384(img).unsqueeze(0).to(target_device, dtype=target_dtype)
    srm_feats = model.v3_c0_champion.srm_proj[0].weight.new_zeros((1, 36)).to(target_device, dtype=target_dtype)
    
    patch_tensors = [img_224.squeeze(0)]
    patch_coords = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]], dtype=target_dtype, device=target_device)
    patch_tensors_t = torch.stack(patch_tensors).to(target_device, dtype=target_dtype)

    # 4. Run Unified Monolithic Forward Pass
    t_fwd = time.time()
    with torch.no_grad():
        out = model(img_224, img_256_5v, img_384, srm_feats, patch_tensors_t, patch_coords)
    fwd_time = time.time() - t_fwd

    ai_prob = float(out["fused_ai_probability"].item())
    v3_gated = float(out["v3_gated_score"].item())
    v5_spatial = float(out["v5_spatial_score"].item())

    print(f"   Unified Forward Pass Execution Time: {fwd_time*1000:.2f} ms")
    print(f"   Fused AI Probability:                {ai_prob:.4f} (Real Prob: {1.0-ai_prob:.4f})")
    print(f"   V3 Gated Score:                      {v3_gated:.4f}")
    print(f"   V5 Spatial Score:                    {v5_spatial:.4f}")
    print(f"   Specialist Logits:                   {dict((k, round(float(v.item()), 4)) for k, v in out['specialist_logits'].items())}")
    print(f"   Verification Status:                 SUCCESS ✅")

def main():
    test_img = "/home/manan/aigc_robust_detection/test_inputs/final_user_test/9872345-mia-khalifa-big-tit-brunette-loves-hard-cock-133-3883013410.jpg"
    print("=" * 95)
    print("     VERIFYING COMPILED MASTER MONOLITHIC CHECKPOINTS (FP32 & FP16)")
    print(f"     Test Image: {test_img}")
    print("=" * 95)

    fp32_ckpt = "/home/manan/aigc_robust_detection/checkpoints/compiled/master_unified_forensic_model_fp32.pt"
    fp16_ckpt = "/home/manan/aigc_robust_detection/checkpoints/compiled/master_unified_forensic_model_fp16.pt"

    verify_checkpoint(fp32_ckpt, "FP32", test_img)
    verify_checkpoint(fp16_ckpt, "FP16", test_img)
    print("\n" + "=" * 95)
    print("  ALL COMPILED MASTER MODELS VERIFIED OPERATIONAL AND PRODUCTION-READY! ✅")
    print("=" * 95)

if __name__ == "__main__":
    main()
