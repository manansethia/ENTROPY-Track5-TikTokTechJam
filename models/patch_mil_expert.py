#!/usr/bin/env python3
"""Patch / Multiple Instance Learning (MIL) Local Expert.
Extracts multiple local crops (random crops, high-frequency patches, edge-heavy patches)
and aggregates local evidence using Gated Attention Multiple Instance Learning (MIL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """Ilse et al. Gated Attention MIL Pooling."""

    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.attn_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attn_u = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, patch_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  B x Num_Patches x In_Dim
        Output: Aggregated Bag Feature (B x In_Dim), Attention Weights (B x Num_Patches)
        """
        # Element-wise product of Tanh and Sigmoid gates
        v = self.attn_v(patch_feats)
        u = self.attn_u(patch_feats)
        attn_scores = self.attn_w(v * u).squeeze(-1) # B x Num_Patches
        attn_weights = F.softmax(attn_scores, dim=-1) # B x Num_Patches
        
        # Weighted sum across all patches
        bag_feature = torch.bmm(attn_weights.unsqueeze(1), patch_feats).squeeze(1) # B x In_Dim
        return bag_feature, attn_weights


class PatchMILExpert(nn.Module):
    """Evaluates multi-view local patch bags using Gated Attention MIL."""

    def __init__(self, patch_dim=768, out_dim=512):
        super().__init__()
        self.mil = GatedAttentionMIL(in_dim=patch_dim, hidden_dim=256)
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, patch_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        patch_embeddings: B x K x patch_dim (K extracted patches per image)
        """
        bag_feat, attn_weights = self.mil(patch_embeddings)
        out = self.proj(bag_feat)
        return out, attn_weights
