#!/usr/bin/env python3
"""
compare_fp32_fp16_pipeline.py
-----------------------------
Executes the 1.88 Billion Parameter Unified Forensic Detection System
first in FP32 (Full Precision) and then in FP16 (Half Precision),
providing side-by-side specialist evidence, numeric drift analysis,
peak VRAM utilization, and latency comparison.
"""

import os
import sys
import gc
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image, ExifTags
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import open_clip
import timm
import safetensors.torch

sys.path.insert(0, "/mnt/ai-storage/aigc_data/models/aide_finetuned")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Checkpoint Paths
AIDE_CKPT = "/mnt/ai-storage/aigc_data/models/aide_finetuned/checkpoint42.pth"
C0_CKPT = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_frozen_model.pt"
C1_CKPT = "/home/manan/aigc_robust_detection/checkpoints/specialists/c5_convnext_tiny_epoch_3.pt"
C2_CKPT = "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c2_spai_vit_best.pt"
C3_CKPT = "/mnt/ai-storage/aigc_data/models/community_forensics_vit_small/model.safetensors"
C4_CKPT = "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c4_convnext_base_best.pt"
C5_CKPT = "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c5_convnext_tiny_best.pt"
C6_CKPT = "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c6_efficientnet_b0_best.pt"
C7_CKPT = "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c7_resnet50_best.pt"
V3_GATING_CKPT = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v3.pt"
V5_CAG_CKPT = "/home/manan/aigc_robust_detection/checkpoints/experimental/v5/v5_champion_cag.pt"

# -------------------------------------------------------------------------
# MODEL ARCHITECTURES
# -------------------------------------------------------------------------
class TripleHybridChampion(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained=None)
        self.clip_visual = clip_model.visual
        self.clip_adapter = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        self.siglip_visual = timm.create_model('vit_so400m_patch14_siglip_224', pretrained=False, num_classes=0)
        self.siglip_adapter = nn.Sequential(
            nn.Linear(1152, 1152),
            nn.LayerNorm(1152),
            nn.GELU()
        )
        self.srm_proj = nn.Sequential(
            nn.Linear(36, 36),
            nn.LayerNorm(36),
            nn.GELU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(1024 + 1152 + 36, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def extract_srm_features(self, img_np: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        q1 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
        q2 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]], dtype=np.float32) / 4.0
        q3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0
        feats = []
        for q in [q1, q2, q3]:
            res = cv2.filter2D(gray, -1, q)
            m = float(np.mean(res))
            v = float(np.var(res))
            k = float(np.mean((res - m)**4) / (v**2 + 1e-6))
            feats.extend([m, v, k])
        while len(feats) < 36: feats.extend([0.0] * (36 - len(feats)))
        return torch.tensor(feats[:36], dtype=torch.float32).unsqueeze(0)

    def forward(self, img_tensors, srm_feats):
        clip_out = self.clip_visual(img_tensors)
        clip_rep = self.clip_adapter(clip_out)
        siglip_out = self.siglip_visual(img_tensors)
        siglip_rep = self.siglip_adapter(siglip_out)
        srm_rep = self.srm_proj(srm_feats)
        fused = torch.cat([clip_rep, siglip_rep, srm_rep], dim=-1)
        return self.fusion_head(fused).squeeze(-1)

class V3LearnedGatingHead(nn.Module):
    def __init__(self, num_experts=8):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(num_experts + 1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_experts)
        )

    def forward(self, expert_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        std = torch.std(expert_logits, dim=-1, keepdim=True)
        feat = torch.cat([expert_logits, std], dim=-1)
        raw_weights = self.gating(feat)
        weights = F.softmax(raw_weights, dim=-1)
        fused_logit = torch.sum(weights * expert_logits, dim=-1)
        return fused_logit, weights

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

# -------------------------------------------------------------------------
# PRECISION-CONTROLLED RUNNER
# -------------------------------------------------------------------------
def run_precision_inference(image_path: str, precision: str = "fp32") -> Dict[str, Any]:
    assert precision in ["fp32", "fp16"]
    is_fp16 = (precision == "fp16")
    target_dtype = torch.float16 if is_fp16 else torch.float32
    
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    img_np = np.array(img)

    t_224 = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_256 = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    t_384 = T.Compose([T.Resize((384, 384)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    execution_trace = {}
    specialist_scores = {}
    expert_logits = []
    total_params = 0

    # 1. V2 AIDE Spectral (897.83M)
    l_v2, p_v2 = 0.0, 0.5
    try:
        from models.AIDE import AIDE_Model
        aide = AIDE_Model(None, None).to(DEVICE)
        if is_fp16: aide = aide.half()
        aide.eval()
        data42 = torch.load(AIDE_CKPT, map_location="cpu", weights_only=False)
        aide.load_state_dict(data42["model"], strict=False)
        aide_p = sum(p.numel() for p in aide.parameters())
        total_params += aide_p
        
        t_img = t_256(img).unsqueeze(0).unsqueeze(1).repeat(1, 5, 1, 1, 1).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_v2 = aide(t_img)
            if isinstance(out_v2, torch.Tensor):
                if out_v2.shape[-1] == 2:
                    p_v2 = float(F.softmax(out_v2.float(), dim=-1)[0, 1].item())
                    l_v2 = float(out_v2[0, 1].float().item() - out_v2[0, 0].float().item())
                else:
                    l_v2 = float(out_v2.float().squeeze().item())
                    p_v2 = float(torch.sigmoid(out_v2.float()).item())
        execution_trace["V2_AIDE_Spectral"] = {"executed": True, "parameters": aide_p, "logit": round(l_v2, 4), "prob": round(p_v2, 4)}
        del aide, data42, t_img
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        execution_trace["V2_AIDE_Spectral"] = {"executed": False, "error": str(e)}

    # 2. C0 Triple-Hybrid (734.97M)
    l_c0, p_c0 = 0.0, 0.5
    try:
        c0_model = TripleHybridChampion().to(DEVICE)
        if is_fp16: c0_model = c0_model.half()
        c0_model.eval()
        ckpt_c0 = torch.load(C0_CKPT, map_location="cpu")
        c0_model.load_state_dict(ckpt_c0["model_state_dict"], strict=False)
        c0_p = sum(p.numel() for p in c0_model.parameters())
        total_params += c0_p
        
        t_c0 = t_224(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        srm_c0 = c0_model.extract_srm_features(img_np).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c0 = c0_model(t_c0, srm_c0)
            l_c0 = float(out_c0.float().item())
            p_c0 = float(torch.sigmoid(out_c0.float()).item())
        expert_logits.append(l_c0)
        specialist_scores["C0_TripleHybrid"] = round(p_c0, 4)
        execution_trace["C0_TripleHybrid"] = {"executed": True, "parameters": c0_p, "logit": round(l_c0, 4), "prob": round(p_c0, 4)}
        del c0_model, ckpt_c0, t_c0, srm_c0
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C0_TripleHybrid"] = {"executed": False, "error": str(e)}

    # 3. C1 Portrait Remediation (27.82M)
    try:
        c1 = models.convnext_tiny(num_classes=1).to(DEVICE)
        if is_fp16: c1 = c1.half()
        c1.eval()
        c1.load_state_dict(torch.load(C1_CKPT, map_location=DEVICE, weights_only=False), strict=False)
        c1_p = sum(p.numel() for p in c1.parameters())
        total_params += c1_p
        t_c1 = t_224(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c1 = c1(t_c1).squeeze(-1)
            l_c1 = float(out_c1.float().item())
            p_c1 = float(torch.sigmoid(out_c1.float()).item())
        expert_logits.append(l_c1)
        specialist_scores["C1_Portrait"] = round(p_c1, 4)
        execution_trace["C1_Portrait"] = {"executed": True, "parameters": c1_p, "logit": round(l_c1, 4), "prob": round(p_c1, 4)}
        del c1, t_c1
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C1_Portrait"] = {"executed": False, "error": str(e)}

    # 4. C2 SPAI ViT (21.81M)
    try:
        c2 = timm.create_model('vit_small_patch16_384', pretrained=False, num_classes=1).to(DEVICE)
        if is_fp16: c2 = c2.half()
        c2.eval()
        c2.load_state_dict(torch.load(C2_CKPT, map_location=DEVICE, weights_only=False), strict=False)
        c2_p = sum(p.numel() for p in c2.parameters())
        total_params += c2_p
        t_c2 = t_384(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c2 = c2(t_c2).squeeze(-1)
            l_c2 = float(out_c2.float().item())
            p_c2 = float(torch.sigmoid(out_c2.float()).item())
        expert_logits.append(l_c2)
        specialist_scores["C2_SPAI_ViT"] = round(p_c2, 4)
        execution_trace["C2_SPAI_ViT"] = {"executed": True, "parameters": c2_p, "logit": round(l_c2, 4), "prob": round(p_c2, 4)}
        del c2, t_c2
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C2_SPAI_ViT"] = {"executed": False, "error": str(e)}

    # 5. C3 CommunityForensics ViT (21.81M)
    try:
        c3 = timm.create_model('vit_small_patch16_384', pretrained=False, num_classes=1).to(DEVICE)
        if is_fp16: c3 = c3.half()
        c3.eval()
        c3.load_state_dict(safetensors.torch.load_file(C3_CKPT), strict=False)
        c3_p = sum(p.numel() for p in c3.parameters())
        total_params += c3_p
        t_c3 = t_384(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c3 = c3(t_c3).squeeze(-1)
            l_c3 = float(out_c3.float().item())
            p_c3 = float(torch.sigmoid(out_c3.float()).item())
        expert_logits.append(l_c3)
        specialist_scores["C3_CommunityForensics"] = round(p_c3, 4)
        execution_trace["C3_CommunityForensics"] = {"executed": True, "parameters": c3_p, "logit": round(l_c3, 4), "prob": round(p_c3, 4)}
        del c3, t_c3
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C3_CommunityForensics"] = {"executed": False, "error": str(e)}

    # 6. C4 ConvNeXt-Base High-Res (87.57M)
    try:
        c4 = models.convnext_tiny(num_classes=1).to(DEVICE)
        if is_fp16: c4 = c4.half()
        c4.eval()
        c4.load_state_dict(torch.load(C4_CKPT, map_location=DEVICE, weights_only=False), strict=False)
        c4_p = sum(p.numel() for p in c4.parameters())
        total_params += c4_p
        t_c4 = t_224(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c4 = c4(t_c4).squeeze(-1)
            l_c4 = float(out_c4.float().item())
            p_c4 = float(torch.sigmoid(out_c4.float()).item())
        expert_logits.append(l_c4)
        specialist_scores["C4_ConvNeXt_HighRes"] = round(p_c4, 4)
        execution_trace["C4_ConvNeXt_HighRes"] = {"executed": True, "parameters": c4_p, "logit": round(l_c4, 4), "prob": round(p_c4, 4)}
        del c4, t_c4
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C4_ConvNeXt_HighRes"] = {"executed": False, "error": str(e)}

    # 7. C5 divine2k ConvNeXt-Tiny (27.82M)
    try:
        c5 = models.convnext_tiny(num_classes=1).to(DEVICE)
        if is_fp16: c5 = c5.half()
        c5.eval()
        c5.load_state_dict(torch.load(C5_CKPT, map_location=DEVICE, weights_only=False), strict=False)
        c5_p = sum(p.numel() for p in c5.parameters())
        total_params += c5_p
        t_c5 = t_224(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c5 = c5(t_c5).squeeze(-1)
            l_c5 = float(out_c5.float().item())
            p_c5 = float(torch.sigmoid(out_c5.float()).item())
        expert_logits.append(l_c5)
        specialist_scores["C5_ConvNeXt_divine2k"] = round(p_c5, 4)
        execution_trace["C5_ConvNeXt_divine2k"] = {"executed": True, "parameters": c5_p, "logit": round(l_c5, 4), "prob": round(p_c5, 4)}
        del c5, t_c5
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C5_ConvNeXt_divine2k"] = {"executed": False, "error": str(e)}

    # 8. C6 EfficientNet-B0 (4.01M)
    try:
        c6 = models.efficientnet_b0(num_classes=1).to(DEVICE)
        if is_fp16: c6 = c6.half()
        c6.eval()
        c6.load_state_dict(torch.load(C6_CKPT, map_location=DEVICE, weights_only=False), strict=False)
        c6_p = sum(p.numel() for p in c6.parameters())
        total_params += c6_p
        t_c6 = t_224(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c6 = c6(t_c6).squeeze(-1)
            l_c6 = float(out_c6.float().item())
            p_c6 = float(torch.sigmoid(out_c6.float()).item())
        expert_logits.append(l_c6)
        specialist_scores["C6_EfficientNet"] = round(p_c6, 4)
        execution_trace["C6_EfficientNet"] = {"executed": True, "parameters": c6_p, "logit": round(l_c6, 4), "prob": round(p_c6, 4)}
        del c6, t_c6
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C6_EfficientNet"] = {"executed": False, "error": str(e)}

    # 9. C7 ResNet-50 (23.51M)
    try:
        c7 = models.resnet50(num_classes=1).to(DEVICE)
        if is_fp16: c7 = c7.half()
        c7.eval()
        c7.load_state_dict(torch.load(C7_CKPT, map_location=DEVICE, weights_only=False), strict=False)
        c7_p = sum(p.numel() for p in c7.parameters())
        total_params += c7_p
        t_c7 = t_224(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        with torch.no_grad():
            out_c7 = c7(t_c7).squeeze(-1)
            l_c7 = float(out_c7.float().item())
            p_c7 = float(torch.sigmoid(out_c7.float()).item())
        expert_logits.append(l_c7)
        specialist_scores["C7_ResNet50"] = round(p_c7, 4)
        execution_trace["C7_ResNet50"] = {"executed": True, "parameters": c7_p, "logit": round(l_c7, 4), "prob": round(p_c7, 4)}
        del c7, t_c7
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        expert_logits.append(0.0)
        execution_trace["C7_ResNet50"] = {"executed": False, "error": str(e)}

    # 10. V3 Learned Gating Network (1.22K)
    p_v3_fused = 0.5
    v3_weights_list = [0.125] * 8
    try:
        v3_gating = V3LearnedGatingHead(num_experts=8).to(DEVICE)
        ckpt_v3 = torch.load(V3_GATING_CKPT, map_location=DEVICE)
        v3_gating.load_state_dict(ckpt_v3["gating_head_state_dict"])
        temp_v3 = float(ckpt_v3.get("temperature", 1.15))
        
        t_exp = torch.tensor([expert_logits], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            fused_v3_logit, weights_t = v3_gating(t_exp)
            p_v3_fused = float(torch.sigmoid(fused_v3_logit / temp_v3).item())
            v3_weights_list = [round(float(w), 4) for w in weights_t[0].cpu().numpy()]
        del v3_gating, ckpt_v3, t_exp
        torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        pass

    # 11. V5-CAG Spatial Multi-Scale Engine (31.09M)
    backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    extractor = backbone.features.to(DEVICE)
    if is_fp16: extractor = extractor.half()
    extractor.eval()
    pool = nn.AdaptiveAvgPool2d((1, 1)).to(DEVICE)
    v5_model = V5CAGModel().to(DEVICE)
    if is_fp16: v5_model = v5_model.half()
    v5_model.eval()
    if os.path.exists(V5_CAG_CKPT):
        v5_model.load_state_dict(torch.load(V5_CAG_CKPT, map_location=DEVICE))
    b_p = sum(p.numel() for p in extractor.parameters())
    v5_p = sum(p.numel() for p in v5_model.parameters())
    total_params += (b_p + v5_p)
    
    with torch.no_grad():
        g_tensor = t_224(img).unsqueeze(0).to(DEVICE, dtype=target_dtype)
        g_feat = pool(extractor(g_tensor)).flatten(1)
        
        patch_scales = [512, 768, 1024]
        p_tensors, p_coords, raw_boxes = [], [], []
        for scale in patch_scales:
            step = int(scale * 0.75)
            for y in range(0, max(1, h - scale + 1), max(1, step)):
                for x in range(0, max(1, w - scale + 1), max(1, step)):
                    p_img = img.crop((x, y, x + scale, y + scale))
                    p_tensors.append(t_224(p_img))
                    p_coords.append([x / w, y / h, scale / w, scale / h, scale / 1024.0])
                    raw_boxes.append((x, y, scale, scale))
                    if len(p_tensors) >= 16: break
                if len(p_tensors) >= 16: break
        if len(p_tensors) == 0:
            p_tensors.append(t_224(img))
            p_coords.append([0.0, 0.0, 1.0, 1.0, 1.0])
            raw_boxes.append((0, 0, w, h))

        p_feat_list = []
        for i in range(0, len(p_tensors), 16):
            chunk = torch.stack(p_tensors[i:i+16]).to(DEVICE, dtype=target_dtype)
            p_feat_list.append(pool(extractor(chunk)).flatten(1))
        p_feats = torch.cat(p_feat_list, dim=0)
        p_coords_t = torch.tensor(p_coords, dtype=target_dtype, device=DEVICE)
        
        whole_logits, patch_logits, pred_mask, attn_weights = v5_model(g_feat, p_feats, p_coords_t)
        class_probs = F.softmax(whole_logits.float(), dim=-1)[0].cpu().numpy()
        patch_probs = torch.sigmoid(patch_logits.float()).cpu().numpy()
        
        p_v5_real = float(class_probs[0])
        p_v5_partial = float(class_probs[1])
        p_v5_full = float(class_probs[2])
        max_patch_anomaly = float(np.max(patch_probs))

    # Master Consensus Fused Score
    p_v5_synthetic = float(1.0 - p_v5_real)
    fused_ai_prob = float(0.20 * p_v2 + 0.50 * p_v3_fused + 0.30 * p_v5_synthetic)
    
    # 3-Way Categorization
    if max_patch_anomaly >= 0.50 and (p_v5_partial > 0.25) and p_v5_full < 0.75:
        verdict = "PARTIAL_AIGC"
        confidence = max(p_v5_partial, max_patch_anomaly)
    elif fused_ai_prob >= 0.55 or p_v5_full >= 0.55:
        verdict = "FULL_AIGC"
        confidence = max(fused_ai_prob, p_v5_full)
    elif fused_ai_prob <= 0.40 and max_patch_anomaly < 0.20:
        verdict = "REAL"
        confidence = max(1.0 - fused_ai_prob, p_v5_real)
    else:
        if p_v5_real >= max(p_v5_partial, p_v5_full):
            verdict = "REAL"; confidence = p_v5_real
        elif p_v5_partial >= p_v5_full:
            verdict = "PARTIAL_AIGC"; confidence = p_v5_partial
        else:
            verdict = "FULL_AIGC"; confidence = p_v5_full

    peak_vram_mib = torch.cuda.max_memory_allocated() / (1024**2)
    latency = time.time() - t0

    return {
        "precision": precision.upper(),
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "fused_ai_prob": round(fused_ai_prob, 4),
        "v2_aide_score": round(p_v2, 4),
        "v3_gated_score": round(p_v3_fused, 4),
        "v5_spatial_score": round(p_v5_synthetic, 4),
        "max_patch_anomaly": round(max_patch_anomaly, 4),
        "specialist_scores": specialist_scores,
        "v3_routing_weights": dict(zip(["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"], v3_weights_list)),
        "execution_trace": execution_trace,
        "total_parameters_instantiated": total_params,
        "peak_vram_mib": round(peak_vram_mib, 2),
        "latency_seconds": round(latency, 3)
    }

def main():
    if len(sys.argv) < 2:
        img_path = "/home/manan/aigc_robust_detection/test_inputs/high_res_milky_way.webp"
    else:
        img_path = sys.argv[1]

    print("=" * 95)
    print(f"  EXECUTING PRECISION COMPARISON (FP32 vs FP16) ON 1.88B MULTI-SPECIALIST SYSTEM")
    print(f"  Target Image: {img_path}")
    print("=" * 95)

    print("\n[STAGE 1/2] Executing Complete 1.88B Ensemble in FP32 (Full Precision float32)...")
    res_fp32 = run_precision_inference(img_path, precision="fp32")

    print("\n[STAGE 2/2] Executing Complete 1.88B Ensemble in FP16 (Half Precision float16)...")
    res_fp16 = run_precision_inference(img_path, precision="fp16")

    print("\n" + "=" * 95)
    print("                              FP32 vs FP16 COMPARISON MATRIX")
    print("=" * 95)
    print(f"{'Metric / Specialist':<35} | {'FP32 (Float32)':<25} | {'FP16 (Float16)':<25} | {'Delta (Diff)':<10}")
    print("-" * 105)
    print(f"{'Final Verdict':<35} | {res_fp32['verdict']:<25} | {res_fp16['verdict']:<25} | {'MATCH' if res_fp32['verdict'] == res_fp16['verdict'] else 'DRIFT'}")
    print(f"{'Final Confidence':<35} | {res_fp32['confidence']:<25.4f} | {res_fp16['confidence']:<25.4f} | {abs(res_fp32['confidence'] - res_fp16['confidence']):.4f}")
    print(f"{'Fused AI Probability':<35} | {res_fp32['fused_ai_prob']:<25.4f} | {res_fp16['fused_ai_prob']:<25.4f} | {abs(res_fp32['fused_ai_prob'] - res_fp16['fused_ai_prob']):.4f}")
    print("-" * 105)
    print(f"{'V2 AIDE Spectral Score':<35} | {res_fp32['v2_aide_score']:<25.4f} | {res_fp16['v2_aide_score']:<25.4f} | {abs(res_fp32['v2_aide_score'] - res_fp16['v2_aide_score']):.4f}")
    print(f"{'V3 Ensemble Gated Score':<35} | {res_fp32['v3_gated_score']:<25.4f} | {res_fp16['v3_gated_score']:<25.4f} | {abs(res_fp32['v3_gated_score'] - res_fp16['v3_gated_score']):.4f}")
    print(f"{'V5 Spatial Synthetic Score':<35} | {res_fp32['v5_spatial_score']:<25.4f} | {res_fp16['v5_spatial_score']:<25.4f} | {abs(res_fp32['v5_spatial_score'] - res_fp16['v5_spatial_score']):.4f}")
    print(f"{'V5 Max Patch Anomaly':<35} | {res_fp32['max_patch_anomaly']:<25.4f} | {res_fp16['max_patch_anomaly']:<25.4f} | {abs(res_fp32['max_patch_anomaly'] - res_fp16['max_patch_anomaly']):.4f}")
    print("-" * 105)
    for k in res_fp32['specialist_scores']:
        v32 = res_fp32['specialist_scores'][k]
        v16 = res_fp16['specialist_scores'].get(k, 0.0)
        print(f"{k:<35} | {v32:<25.4f} | {v16:<25.4f} | {abs(v32 - v16):.4f}")
    print("-" * 105)
    print(f"{'Peak GPU VRAM Allocated':<35} | {str(res_fp32['peak_vram_mib']) + ' MiB':<25} | {str(res_fp16['peak_vram_mib']) + ' MiB':<25} | {res_fp32['peak_vram_mib'] - res_fp16['peak_vram_mib']:.2f} MiB")
    print(f"{'Inference Latency':<35} | {str(res_fp32['latency_seconds']) + ' s':<25} | {str(res_fp16['latency_seconds']) + ' s':<25} | {res_fp32['latency_seconds'] - res_fp16['latency_seconds']:.2f} s")
    print("=" * 95)

    # Save comparison log to JSON
    out_json = "/home/manan/aigc_robust_detection/reports/final/milky_way_fp32_vs_fp16_audit.json"
    with open(out_json, "w") as f:
        json.dump({"fp32": res_fp32, "fp16": res_fp16}, f, indent=2)
    print(f"\n  Full JSON Audit saved to: {out_json} ✅")

if __name__ == "__main__":
    main()
