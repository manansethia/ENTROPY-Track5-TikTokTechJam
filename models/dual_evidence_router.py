#!/usr/bin/env python3
"""Dual-Evidence Reliability Router.
Separates incoming multimodal features into two explicit competing channels:
1. AI Evidence Channel: Accumulates generative artifacts, unnatural frequency decay, edge anomalies.
2. Real Evidence Channel: Accumulates authentic camera optics, natural sensor noise, photorealistic physics.
Calculates final log-likelihood ratio with calibrated Bayesian prior shift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEvidenceReliabilityRouter(nn.Module):
    """Multi-view Dual-Evidence Reliability Gating and Classification Engine."""

    def __init__(
        self,
        d_semantic=1152,   # SigLIP-SO400M
        d_geometry=1024,   # DINOv2-Large
        d_clip=1024,       # CLIP ViT-L/14
        d_frequency=201,   # 2D FFT Spectral Engine
        d_edge=256,        # E²GenF Edge Specialist
        d_patch_mil=512,   # Patch / MIL Bag Feature
        embed_dim=256,
    ):
        super().__init__()
        # 1. Feature Projection Layers
        self.proj_semantic = nn.Sequential(nn.Linear(d_semantic, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_geometry = nn.Sequential(nn.Linear(d_geometry, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_clip = nn.Sequential(nn.Linear(d_clip, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_freq = nn.Sequential(nn.Linear(d_frequency, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_edge = nn.Sequential(nn.Linear(d_edge, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.proj_mil = nn.Sequential(nn.Linear(d_patch_mil, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())

        total_in = embed_dim * 6

        # 2. Reliability Router Gates (6 weights summing to 1.0)
        self.router = nn.Sequential(
            nn.Linear(total_in, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6),
        )

        # 3. Dual-Evidence Competing Branches
        # Branch A: AI Synthesis Evidence Stream
        self.ai_evidence_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Branch B: Real Authenticity Evidence Stream
        self.real_evidence_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        f_sem: torch.Tensor,
        f_geo: torch.Tensor,
        f_clip: torch.Tensor,
        f_freq: torch.Tensor,
        f_edge: torch.Tensor,
        f_mil: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          - Final 2-class Logits: (B x 2)
          - Gate Weights: (B x 6) [Semantic, Geometry, CLIP, Frequency, Edge, MIL]
          - AI Evidence Score: (B x 1)
          - Real Evidence Score: (B x 1)
        """
        e_sem = self.proj_semantic(f_sem)
        e_geo = self.proj_geometry(f_geo)
        e_clp = self.proj_clip(f_clip)
        e_frq = self.proj_freq(f_freq)
        e_edg = self.proj_edge(f_edge)
        e_mil = self.proj_mil(f_mil)

        stacked_features = torch.cat([e_sem, e_geo, e_clp, e_frq, e_edg, e_mil], dim=-1) # B x (6 * embed_dim)

        # Dynamic Softmax Routing Gates
        gate_logits = self.router(stacked_features)
        gate_weights = F.softmax(gate_logits, dim=-1) # B x 6

        # Dynamically Fused Unified Representation
        expert_stack = torch.stack([e_sem, e_geo, e_clp, e_frq, e_edg, e_mil], dim=1) # B x 6 x embed_dim
        fused_expert = (expert_stack * gate_weights.unsqueeze(-1)).sum(dim=1) # B x embed_dim

        # Competing Dual Evidence Scores
        score_ai = self.ai_evidence_head(fused_expert)      # Evidence that image is AI
        score_real = self.real_evidence_head(fused_expert)  # Evidence that image is Real

        # Output Logits [logit_real, logit_fake]
        logits = torch.cat([score_real, score_ai], dim=-1)
        return logits, gate_weights, score_ai, score_real
