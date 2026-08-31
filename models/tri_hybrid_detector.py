import torch
import torch.nn as nn
from transformers import AutoModel
import open_clip
import timm

from .srm_filters import WaveletResidualBlock


class MasterEnsembleDetector(nn.Module):
    """Tri-stream semantic/frequency AIGC detector.

    Stream A: frozen CLIP ViT-L/14 semantic features.
    Stream B: frozen SigLIP/SigLIP2 semantic features.
    Stream C: trainable SRM + Haar detail + ConvNeXt-Tiny frequency features.

    The learned gate produces three weights that sum to one, allowing the model
    to reduce reliance on frequency evidence when blur/compression destroys it.
    """

    def __init__(
        self,
        clip_name="ViT-L-14",
        clip_pretrained="openai",
        siglip_name="google/siglip-base-patch16-224",
        convnext_name="convnext_tiny",
        proj_dim=256,
        dropout_prob=0.30,
    ):
        super().__init__()

        # Stream A: CLIP Foundation Vision Encoder (Frozen)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_name, pretrained=clip_pretrained
        )
        self.clip_encoder = self.clip_model.visual
        for p in self.clip_encoder.parameters():
            p.requires_grad = False

        # Detect CLIP feature dimension
        clip_dim = getattr(self.clip_model.visual, "output_dim", 768)

        # Stream B: SigLIP / SigLIP2 Vision Encoder (Frozen)
        self.siglip_model = AutoModel.from_pretrained(siglip_name)
        self.siglip_encoder = (
            self.siglip_model.vision_model
            if hasattr(self.siglip_model, "vision_model")
            else self.siglip_model
        )
        for p in self.siglip_encoder.parameters():
            p.requires_grad = False

        # Detect SigLIP feature dimension
        if hasattr(self.siglip_model.config, "vision_config"):
            siglip_dim = self.siglip_model.config.vision_config.hidden_size
        else:
            siglip_dim = getattr(self.siglip_model.config, "hidden_size", 768)

        # Stream C: High-pass SRM + 2D Haar Wavelet Residual + ConvNeXt Trunk (Trainable)
        self.residual_extractor = WaveletResidualBlock()
        self.freq_encoder = timm.create_model(
            convnext_name, pretrained=True, in_chans=9, num_classes=0
        )
        freq_dim = self.freq_encoder.num_features

        # Projection Layers
        self.proj_clip = nn.Linear(clip_dim, proj_dim)
        self.proj_siglip = nn.Linear(siglip_dim, proj_dim)
        self.proj_freq = nn.Linear(freq_dim, proj_dim)

        # Reliability-Aware Gating Network
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )

        # Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 1),
        )

    def extract_features(self, x_clip, x_siglip, x_raw):
        """Extract multi-stream embeddings before gating/fusion."""
        with torch.no_grad():
            f_clip = self.clip_encoder(x_clip)
            if isinstance(f_clip, tuple):
                f_clip = f_clip[0]

            siglip_out = self.siglip_encoder(pixel_values=x_siglip)
            f_siglip = getattr(siglip_out, "pooler_output", None)
            if f_siglip is None:
                # If pooler_output is not present, use mean pooled last hidden state
                f_siglip = siglip_out.last_hidden_state.mean(dim=1)

        f_clip = self.proj_clip(f_clip)
        f_siglip = self.proj_siglip(f_siglip)

        f_res = self.residual_extractor(x_raw)
        f_freq = self.freq_encoder(f_res)
        f_freq = self.proj_freq(f_freq)

        return f_clip, f_siglip, f_freq

    def forward(self, x_clip, x_siglip, x_raw, return_gate=False, return_streams=False):
        f_clip, f_siglip, f_freq = self.extract_features(x_clip, x_siglip, x_raw)

        f_cat = torch.cat([f_clip, f_siglip, f_freq], dim=-1)
        weights = self.gate(f_cat)

        f_fused = (
            weights[:, 0:1] * f_clip
            + weights[:, 1:2] * f_siglip
            + weights[:, 2:3] * f_freq
        )
        logits = self.classifier(f_fused).squeeze(-1)

        if return_streams:
            return logits, weights, (f_clip, f_siglip, f_freq)
        if return_gate:
            return logits, weights
        return logits

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}
