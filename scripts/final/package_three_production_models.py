#!/usr/bin/env python3
"""
package_three_production_models.py
----------------------------------
Packages all 11 genuine trained historical specialist models and the trained
Master Intelligent Fusion Head into THREE production candidate model formats:
  1. FP32 Master Model: checkpoints/production_candidate/master_intelligent_forensic_model_fp32.pt (~6.8 GB)
  2. FP16 Master Model: checkpoints/production_candidate/master_intelligent_forensic_model_fp16.pt (~3.4 GB)
  3. FP8 / INT8 Quantized Model: checkpoints/production_candidate/master_intelligent_forensic_model_fp8.pt (~1.7 GB)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import open_clip
import timm
import safetensors.torch
import importlib.util

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.master_intelligent_fusion_head import MasterIntelligentFusionHead
from scripts.final.compile_master_unified_model import (
    MasterUnifiedForensicModel,
    TripleHybridChampion,
    V3LearnedGatingHead,
    V5CAGModel,
    load_trained_weights_
)

# -------------------------------------------------------------------------
# Complete End-to-End Master Intelligent Forensic Architecture
# -------------------------------------------------------------------------
class CompleteMasterIntelligentForensicModel(nn.Module):
    """
    Unified Intelligent Forensic Neural Model containing all 11 specialist trunks
    and the trained Master Intelligent Cross-Attention Fusion Head.
    """
    def __init__(self):
        super().__init__()
        # 11 Specialist Trunks
        self.specialists = MasterUnifiedForensicModel()
        # Master Intelligent Cross-Attention Fusion Head
        self.master_head = MasterIntelligentFusionHead()

    def forward(
        self,
        img_224: torch.Tensor,
        img_256_spectral_5v: torch.Tensor,
        img_384: torch.Tensor,
        srm_features: torch.Tensor,
        patch_tensors_224: torch.Tensor,
        patch_coords: torch.Tensor
    ) -> Dict[str, Any]:
        # 1. Forward through Specialist Trunks
        spec_out = self.specialists(img_224, img_256_spectral_5v, img_384, srm_features, patch_tensors_224, patch_coords)
        
        # 2. Extract Multi-Modal Evidence Vectors
        spec_logits = torch.stack([
            spec_out["specialist_logits"]["C0"],
            spec_out["specialist_logits"]["C1"],
            spec_out["specialist_logits"]["C2"],
            spec_out["specialist_logits"]["C3"],
            spec_out["specialist_logits"]["C4"],
            spec_out["specialist_logits"]["C5"],
            spec_out["specialist_logits"]["C6"],
            spec_out["specialist_logits"]["C7"]
        ], dim=-1)

        v2_score = spec_out["v2_spectral_score"].view(-1, 1)
        v3_score = spec_out["v3_gated_score"].view(-1, 1)
        
        target_dtype = img_224.dtype
        # V5 class probabilities
        p_real = spec_out["real_probability"].view(-1, 1).to(target_dtype)
        p_partial = spec_out["partial_ai_probability"].view(-1, 1).to(target_dtype)
        p_full = spec_out["full_aigc_probability"].view(-1, 1).to(target_dtype)
        v5_probs = torch.cat([p_real, p_partial, p_full], dim=-1)

        # Patch stats
        max_anom = spec_out["patch_anomalies"].max().view(-1, 1).to(target_dtype)
        mean_anom = spec_out["patch_anomalies"].mean().view(-1, 1).to(target_dtype)
        patch_stats = torch.cat([max_anom, mean_anom], dim=-1)

        # Spatial Embedding from V5 Backbone (256D)
        g_feat = self.specialists.v5_pool(self.specialists.v5_backbone(img_224)).flatten(1)
        p_feats = self.specialists.v5_pool(self.specialists.v5_backbone(patch_tensors_224)).flatten(1)
        pos_emb = self.specialists.v5_cag_head.pos_mlp(patch_coords)
        combined = torch.cat([g_feat.expand(p_feats.shape[0], -1), p_feats, pos_emb], dim=-1)
        fused_spatial = self.specialists.v5_cag_head.fusion_mlp(combined)
        global_spatial = fused_spatial.mean(dim=0, keepdim=True).to(target_dtype) # (1, 256)

        # 3. Forward through Master Intelligent Head
        intel_out = self.master_head(
            specialist_logits=spec_logits,
            v2_spectral_score=v2_score,
            v3_gated_score=v3_score,
            v5_spatial_probs=v5_probs,
            v5_patch_stats=patch_stats,
            spatial_embedding=global_spatial
        )

        return {
            "verdict_logits": intel_out["class_logits"],
            "calibrated_probs": intel_out["calibrated_probs"],
            "real_probability": intel_out["real_prob"],
            "partial_ai_probability": intel_out["partial_ai_prob"],
            "full_aigc_probability": intel_out["full_aigc_prob"],
            "uncertainty": intel_out["uncertainty"],
            "segmentation_heatmap": intel_out["segmentation_heatmap"],
            "patch_anomalies": spec_out["patch_anomalies"],
            "v2_spectral_score": spec_out["v2_spectral_score"],
            "v3_gated_score": spec_out["v3_gated_score"],
            "v3_routing_weights": spec_out["v3_routing_weights"],
            "specialist_logits": spec_out["specialist_logits"]
        }

def quantize_state_dict_int8(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Uniform dynamic 8-bit quantization for weights while keeping biases and norms in FP16.
    """
    quantized_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.numel() > 1024 and "weight" in k:
            max_val = v.abs().max()
            scale = max_val / 127.0 if max_val > 0 else 1.0
            q_tensor = (v / scale).round().clamp(-128, 127).to(torch.int8)
            quantized_sd[k] = {
                "qweight": q_tensor,
                "scale": scale,
                "is_quantized": True
            }
        else:
            quantized_sd[k] = v.half() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
    return quantized_sd

def main():
    print("=" * 105)
    print("  PACKAGING THREE FINAL MASTER INTELLIGENT FORENSIC MODELS (FP32, FP16, FP8)")
    print("=" * 105)
    
    t_start = time.time()
    out_dir = "/home/manan/aigc_robust_detection/checkpoints/production_candidate"
    os.makedirs(out_dir, exist_ok=True)

    head_ckpt_path = os.path.join(out_dir, "master_intelligent_head.pt")
    fp32_path = os.path.join(out_dir, "master_intelligent_forensic_model_fp32.pt")
    fp16_path = os.path.join(out_dir, "master_intelligent_forensic_model_fp16.pt")
    fp8_path = os.path.join(out_dir, "master_intelligent_forensic_model_fp8.pt")

    # Step 1: Instantiate & Load Model
    print("\n[STEP 1/4] Instantiating Complete Master Intelligent Model Architecture...")
    model = CompleteMasterIntelligentForensicModel()
    load_trained_weights_(model.specialists)
    
    if os.path.exists(head_ckpt_path):
        print(f"  Loading trained Master Intelligent Head -> {head_ckpt_path}...")
        hd = torch.load(head_ckpt_path, map_location="cpu")
        model.master_head.load_state_dict(hd["head_state_dict"])
    else:
        print("  Warning: Master Head Checkpoint not found, initializing base fusion weights.")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Master Model Parameters: {total_params:,} (~{total_params/1e9:.3f} Billion)")

    # Step 2: Export FP32 Model
    print(f"\n[STEP 2/4] Serializing FP32 Master Model Checkpoint -> {fp32_path}...")
    t0 = time.time()
    torch.save({
        "model_state_dict": model.state_dict(),
        "total_parameters": total_params,
        "precision": "FP32",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": "CompleteMasterIntelligentForensicModel"
    }, fp32_path)
    size_fp32_gb = os.path.getsize(fp32_path) / (1024**3)
    print(f"  FP32 Checkpoint Saved: {size_fp32_gb:.2f} GB in {time.time()-t0:.2f}s ✅")

    # Step 3: Export FP16 Model
    print(f"\n[STEP 3/4] Converting to Half Precision & Serializing FP16 Model -> {fp16_path}...")
    t0 = time.time()
    model_fp16 = model.half()
    torch.save({
        "model_state_dict": model_fp16.state_dict(),
        "total_parameters": total_params,
        "precision": "FP16",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": "CompleteMasterIntelligentForensicModel"
    }, fp16_path)
    size_fp16_gb = os.path.getsize(fp16_path) / (1024**3)
    print(f"  FP16 Checkpoint Saved: {size_fp16_gb:.2f} GB in {time.time()-t0:.2f}s ✅")

    # Step 4: Export FP8 / INT8 Quantized Model
    print(f"\n[STEP 4/4] Applying 8-Bit Uniform Quantization & Serializing FP8 Model -> {fp8_path}...")
    t0 = time.time()
    q_sd = quantize_state_dict_int8(model.state_dict())
    torch.save({
        "model_state_dict": q_sd,
        "total_parameters": total_params,
        "precision": "FP8_INT8",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": "CompleteMasterIntelligentForensicModel"
    }, fp8_path)
    size_fp8_gb = os.path.getsize(fp8_path) / (1024**3)
    print(f"  FP8/INT8 Checkpoint Saved: {size_fp8_gb:.2f} GB in {time.time()-t0:.2f}s ✅")

    total_time = time.time() - t_start
    print("\n" + "=" * 105)
    print("                       PRODUCTION MODELS PACKAGING REPORT")
    print("=" * 105)
    print(f"  Architecture:                CompleteMasterIntelligentForensicModel")
    print(f"  Total Parameters:            {total_params:,} (~{total_params/1e9:.3f} Billion)")
    print(f"  FP32 Master Model:           {fp32_path} ({size_fp32_gb:.2f} GB)")
    print(f"  FP16 Master Model:           {fp16_path} ({size_fp16_gb:.2f} GB)")
    print(f"  FP8 / INT8 Master Model:     {fp8_path} ({size_fp8_gb:.2f} GB)")
    print(f"  Total Packaging Time:        {total_time:.2f} seconds")
    print("=" * 105)

if __name__ == "__main__":
    main()
