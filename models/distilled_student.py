"""
AetherForensics — Amalgamated Single-Student Forensics Architecture
Distills a 4-Stream Multi-Paradigm Ensemble (722M params) into a single unified backbone (<86M params).
Supports FP32, FP16, and INT8 Post-Training Quantization for edge/mobile deployment.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ConvNextV2Model



class SingleStudentForensicDetector(nn.Module):
    """
    Unified Single-Student Forensics Detector.
    Backbone: ConvNeXt-V2-Tiny (28M) or SigLIP-Base (86M).
    Embeds Multi-Teacher Knowledge:
      - Spatial Token Variance (from SigLIP)
      - Semantic Macro-Coherence (from CLIP ViT-L/14)
      - 3D Structural Geometry (from DINOv2)
      - Continuous Convolution Inductive Bias (from ConvNeXt-V2)
    """
    def __init__(
        self,
        student_dim: int = 768,
        dim_siglip: int = 768,
        dim_clip: int = 1024,
        dim_dino: int = 1024,
        dim_convnext: int = 768,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.student_dim = student_dim

        # 1. Primary Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(student_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 2. Multi-Teacher Feature Mimic Projectors (Active during Distillation)
        self.proj_siglip = nn.Sequential(
            nn.Linear(student_dim, dim_siglip),
            nn.LayerNorm(dim_siglip)
        )
        self.proj_clip = nn.Sequential(
            nn.Linear(student_dim, dim_clip),
            nn.LayerNorm(dim_clip)
        )
        self.proj_dino = nn.Sequential(
            nn.Linear(student_dim, dim_dino),
            nn.LayerNorm(dim_dino)
        )
        self.proj_convnext = nn.Sequential(
            nn.Linear(student_dim, dim_convnext),
            nn.LayerNorm(dim_convnext)
        )

    def forward(self, f_student: torch.Tensor, return_projections: bool = False):
        """
        Forward pass from student feature representation.
        Args:
            f_student: (B, student_dim) pooled feature tensor.
            return_projections: If True, returns feature projections for teacher MSE matching.
        """
        # Normalize
        f_norm = f_student / f_student.norm(dim=-1, keepdim=True)
        logits = self.classifier(f_norm)

        if not return_projections:
            return logits

        # Multi-Teacher Mimic Projections
        p_siglip = self.proj_siglip(f_norm)
        p_clip = self.proj_clip(f_norm)
        return logits, (p_siglip, p_clip, p_dino, p_convnext)


class EndToEndNativeStudentDetector(nn.Module):
    """
    End-to-End Native Forensic Detector for ONNX/CoreML/DirectML/TensorRT export.
    Accepts raw pixel input tensor (B, 3, 224, 224) and outputs class logits (B, 2).
    """
    def __init__(self, backbone_dir="/mnt/ai-storage/aigc_data/models/convnext_v2_tiny"):
        super().__init__()
        if os.path.isdir(backbone_dir):
            self.backbone = ConvNextV2Model.from_pretrained(backbone_dir)
        else:
            from transformers import ConvNextV2Config
            cfg = ConvNextV2Config(hidden_sizes=[96, 192, 384, 768])
            self.backbone = ConvNextV2Model(cfg)
        self.head = SingleStudentForensicDetector(student_dim=768)

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.backbone(pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            f = outputs.pooler_output
        else:
            f = outputs.last_hidden_state.mean(dim=[-2, -1])
        f = f.reshape(-1, 768)
        logits = self.head(f, return_projections=False)
        return logits



class DistillationLoss(nn.Module):
    """
    Combined Hard Label + Soft Teacher Logit KL + Multi-Teacher Feature Representation Loss.
    """
    def __init__(
        self,
        temperature: float = 2.0,
        alpha_ce: float = 0.5,
        alpha_kd: float = 0.3,
        alpha_feat: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.alpha_feat = alpha_feat

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_projections: tuple,
        teacher_features: tuple,
        targets: torch.Tensor
    ):
        # 1. Hard Cross-Entropy Loss
        loss_ce = self.ce_loss(student_logits, targets)

        # 2. Soft Dark-Knowledge Logit Matching (KL Divergence with Temperature)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss_kd = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        # 3. Multi-Teacher Intermediate Feature Mimic Loss
        p_siglip, p_clip, p_dino, p_convnext = student_projections
        t_siglip, t_clip, t_dino, t_convnext = teacher_features

        loss_feat = (
            self.mse_loss(p_siglip, t_siglip) +
            self.mse_loss(p_clip, t_clip) +
            self.mse_loss(p_dino, t_dino) +
            self.mse_loss(p_convnext, t_convnext)
        ) / 4.0

        total_loss = (
            self.alpha_ce * loss_ce +
            self.alpha_kd * loss_kd +
            self.alpha_feat * loss_feat
        )

        return total_loss, {
            "loss_ce": loss_ce.item(),
            "loss_kd": loss_kd.item(),
            "loss_feat": loss_feat.item()
        }
