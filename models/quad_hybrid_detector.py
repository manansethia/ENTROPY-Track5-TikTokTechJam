"""Quad-Hybrid Multi-Paradigm AIGC Detector Architecture.
Combines 4 orthogonal vision paradigms:
1. Spatial Boundary ViT: Google SigLIP-Base-224 (768-d)
2. Semantic Composition ViT: OpenAI CLIP ViT-L/14 (1024-d)
3. 3D Geometric Depth ViT: Meta DINOv2-Large (1024-d)
4. Pure Continuous Convolution CNN: Meta ConvNeXt-V2-Tiny (768-d)
Controlled by a 4-Way Dynamic Softmax Router. Total Params: ~722M (<2.0B).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadHybridGatingHead(nn.Module):
    def __init__(
        self,
        dim_siglip: int = 768,
        dim_clip: int = 1024,
        dim_dinov2: int = 1024,
        dim_convnext: int = 768,
        proj_dim: int = 512,
        dropout_prob: float = 0.25,
    ):
        super().__init__()
        # 1. Orthogonal Projections
        self.proj_siglip = nn.Sequential(
            nn.Linear(dim_siglip, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(dim_clip, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.proj_dinov2 = nn.Sequential(
            nn.Linear(dim_dinov2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )
        self.proj_convnext = nn.Sequential(
            nn.Linear(dim_convnext, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        )

        # 2. Dynamic 4-Way Softmax Gating Router
        self.router = nn.Sequential(
            nn.Linear(proj_dim * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 4),
        )

        # 3. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 2),
        )

    def forward(self, feat_siglip, feat_clip, feat_dinov2, feat_convnext):
        feat_siglip = feat_siglip.reshape(-1, 768)
        feat_clip = feat_clip.reshape(-1, 1024)
        feat_dinov2 = feat_dinov2.reshape(-1, 1024)
        feat_convnext = feat_convnext.reshape(-1, 768)

        p_siglip = self.proj_siglip(feat_siglip)
        p_clip = self.proj_clip(feat_clip)
        p_dinov2 = self.proj_dinov2(feat_dinov2)
        p_convnext = self.proj_convnext(feat_convnext)

        concat_all = torch.cat([p_siglip, p_clip, p_dinov2, p_convnext], dim=-1)
        gates = F.softmax(self.router(concat_all), dim=-1)  # [B, 4]

        # Dynamic Weighted Fusion
        g_s = gates[:, 0:1]
        g_c = gates[:, 1:2]
        g_d = gates[:, 2:3]
        g_x = gates[:, 3:4]

        fused = g_s * p_siglip + g_c * p_clip + g_d * p_dinov2 + g_x * p_convnext
        logits = self.classifier(fused)
        return logits, gates
