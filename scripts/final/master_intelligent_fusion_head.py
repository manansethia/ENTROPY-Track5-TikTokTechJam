#!/usr/bin/env python3
"""
master_intelligent_fusion_head.py
---------------------------------
Master Intelligent Forensic Fusion Head that learns to combine representations from:
  1. V2 AIDE Spectral Trunk (Frequency domain / High-pass)
  2. V3 Specialists C0–C7 (Triple-Hybrid, Portraits, Frequency ViT, Community, High-Res, Generalist, Edge, Deep)
  3. V3 Learned Gating Network (Dynamic expert weighting)
  4. V5-CAG Spatial Engine (Multi-scale patch attention & spatial feature maps)

Outputs:
  - 3-Way Classification Logits: [P(Real), P(Partial-AI), P(Full-AIGC)]
  - Calibrated Confidence Score
  - Continuous 64x64 Spatial Anomaly Segmentation Map
  - Patch Anomaly Attribution Vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any

class MasterIntelligentFusionHead(nn.Module):
    def __init__(
        self,
        specialist_dim: int = 8,
        spatial_feat_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 3
    ):
        super().__init__()
        
        # 1. Specialist & Meta-Feature Encoder
        # Inputs: 8 specialist logits + 1 std + 1 V2 spectral score + 1 V3 gated score + 3 V5 class probs + 2 patch stats = 16D
        self.meta_encoder = nn.Sequential(
            nn.Linear(specialist_dim + 8, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        # 2. Spatial Representation Projector
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        # 3. Cross-Attention Forensic Fusion Core
        self.cross_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(128)

        # 4. Joint Forensic Trunk
        self.joint_trunk = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )

        # 5. Output Heads
        # Head A: 3-Way Verdict Classifier (Real, Partial-AI, Full-AIGC)
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # Head B: Calibrated Uncertainty / Temperature Estimator
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Head C: Spatial Segmentation Heatmap Decoder (64x64)
        self.heatmap_decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, 512),
            nn.GELU(),
            nn.Linear(512, 64 * 64),
            nn.Sigmoid()
        )

    def forward(
        self,
        specialist_logits: torch.Tensor,     # (B, 8)
        v2_spectral_score: torch.Tensor,     # (B, 1)
        v3_gated_score: torch.Tensor,        # (B, 1)
        v5_spatial_probs: torch.Tensor,      # (B, 3) [real, partial, full]
        v5_patch_stats: torch.Tensor,        # (B, 2) [max_anomaly, mean_anomaly]
        spatial_embedding: torch.Tensor      # (B, 256)
    ) -> Dict[str, torch.Tensor]:
        dtype = self.meta_encoder[0].weight.dtype
        device = self.meta_encoder[0].weight.device

        specialist_logits = specialist_logits.to(device=device, dtype=dtype)
        v2_spectral_score = v2_spectral_score.to(device=device, dtype=dtype)
        v3_gated_score = v3_gated_score.to(device=device, dtype=dtype)
        v5_spatial_probs = v5_spatial_probs.to(device=device, dtype=dtype)
        v5_patch_stats = v5_patch_stats.to(device=device, dtype=dtype)
        spatial_embedding = spatial_embedding.to(device=device, dtype=dtype)

        B = specialist_logits.shape[0]
        
        # Calculate specialist disagreement std
        spec_std = torch.std(specialist_logits, dim=-1, keepdim=True)
        
        # Construct composite forensic descriptor (B, 16)
        meta_features = torch.cat([
            specialist_logits,
            spec_std,
            v2_spectral_score,
            v3_gated_score,
            v5_spatial_probs,
            v5_patch_stats
        ], dim=-1)

        # Encode Meta & Spatial Features
        meta_rep = self.meta_encoder(meta_features).unsqueeze(1)    # (B, 1, 128)
        spat_rep = self.spatial_proj(spatial_embedding).unsqueeze(1) # (B, 1, 128)

        # Multi-Head Cross Attention
        attn_out, _ = self.cross_attn(query=meta_rep, key=spat_rep, value=spat_rep)
        fused_meta = self.fusion_norm(meta_rep + attn_out).squeeze(1) # (B, 128)
        spat_flat = spat_rep.squeeze(1)                              # (B, 128)

        # Joint Fusion
        joint_input = torch.cat([fused_meta, spat_flat], dim=-1)     # (B, 256)
        joint_rep = self.joint_trunk(joint_input)                    # (B, 128)

        # Predictions
        class_logits = self.classifier(joint_rep)                    # (B, 3)
        uncertainty = self.uncertainty_head(joint_rep)               # (B, 1)
        seg_mask = self.heatmap_decoder(joint_rep).view(B, 1, 64, 64) # (B, 1, 64, 64)

        # Temperature-calibrated probabilities
        temperature = 1.0 + uncertainty * 0.5
        calibrated_probs = F.softmax(class_logits / temperature, dim=-1)

        return {
            "class_logits": class_logits,
            "calibrated_probs": calibrated_probs,
            "real_prob": calibrated_probs[:, 0],
            "partial_ai_prob": calibrated_probs[:, 1],
            "full_aigc_prob": calibrated_probs[:, 2],
            "uncertainty": uncertainty.squeeze(-1),
            "segmentation_heatmap": seg_mask
        }
