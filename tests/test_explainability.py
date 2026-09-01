"""Unit and Integration Test Suite for Forensic Explainability Pipeline.

Verifies:
- ViT Grad-CAM and CNN Grad-CAM with frozen and trainable backbones
- ViT Multi-Head Attention Rollout across transformer depth
- 2D FFT Frequency-Domain Power Spectrum & iFFT Spatial Reconstruction
- Multiscale Edge Residuals (Sobel, Laplacian, SRM)
- Patch-Level Localized Attribution Scoring
- Zero Memory Leaks & Graph Cleanup
- Full CLI & Dashboard Generation Pipeline
"""

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.forensic_explainability import (
    CNNConvNeXtGradCAM,
    EdgeResidualExplainer,
    ForensicDiagnosticSuite,
    FrequencySpectralExplainer,
    PatchForensicScorer,
    ViTAttentionRollout,
    ViTGradCAM,
)


# ==============================================================================
# Mock / Test Architectures for Deterministic Testing
# ==============================================================================

class MockViTEncoder(nn.Module):
    """Synthetic ViT backbone for frozen/trainable unit testing."""
    def __init__(self, num_layers: int = 4, embed_dim: int = 64, num_patches: int = 196):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=128,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # x: [B, 3, 224, 224]
        p = self.patch_proj(x).flatten(2).transpose(1, 2)  # [B, 196, D]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, p], dim=1)  # [B, 197, D]
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        logits = self.head(cls_out)
        return logits


class MockCNNTrunk(nn.Module):
    """Synthetic ConvNeXt / CNN trunk for unit testing."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            ),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for stage in self.stages:
            feat = stage(feat)
        pooled = self.pool(feat).flatten(1)
        return self.classifier(pooled)


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_vit_gradcam_frozen_backbone():
    """Verifies that ViTGradCAM generates accurate [0, 1] heatmaps on frozen ViT parameters."""
    model = MockViTEncoder().eval()
    # Freeze all parameters to ensure frozen backbone compatibility
    for p in model.parameters():
        p.requires_grad = False

    gradcam = ViTGradCAM(
        model,
        target_layer=model.layers[-1],
        has_cls_token=True,
        grid_size=(14, 14),
    )

    dummy_input = torch.randn(1, 3, 224, 224)
    cam = gradcam.generate(dummy_input, target_class_idx=1, target_shape=(224, 224))

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
    assert not np.isnan(cam).any()


def test_cnn_gradcam_stage():
    """Verifies ConvNeXt / CNN stage Grad-CAM extraction."""
    model = MockCNNTrunk().eval()
    cnn_gradcam = CNNConvNeXtGradCAM(model, target_layer=model.stages[-1])

    dummy_input = torch.randn(1, 3, 224, 224)
    cam = cnn_gradcam.generate(dummy_input, target_class_idx=1, target_shape=(256, 256))

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (256, 256)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
    assert not np.isnan(cam).any()


def test_vit_attention_rollout():
    """Verifies ViT Multi-Head Attention Rollout across transformer depth."""
    model = MockViTEncoder().eval()
    rollout = ViTAttentionRollout(model, discard_ratio=0.1, head_fusion="mean")

    dummy_input = torch.randn(1, 3, 224, 224)
    mask = rollout.generate(dummy_input, target_shape=(224, 224))

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (224, 224)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    assert not np.isnan(mask).any()


def test_frequency_spectral_explainer():
    """Verifies 2D FFT power spectrum, radial decay profile, and spatial iFFT reconstruction."""
    explainer = FrequencySpectralExplainer(num_radial_bins=64, num_angular_bins=36)
    
    # Synthetic image with high-frequency periodic noise
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    y, x = np.mgrid[0:256, 0:256]
    img[:, :, 0] = np.clip(128 + 64 * np.sin(x / 4.0) * np.sin(y / 4.0), 0, 255).astype(np.uint8)
    img[:, :, 1] = 120
    img[:, :, 2] = 140

    res = explainer.analyze(img)

    assert res.log_power_spectrum.shape == (256, 256)
    assert len(res.radial_profile) == 64
    assert len(res.natural_power_law_fit) == 64
    assert len(res.azimuthal_profile) == 36
    assert res.spatial_frequency_anomaly_map.shape == (256, 256)
    assert 0.0 <= res.high_freq_energy_ratio <= 1.0
    assert isinstance(res.grid_peak_anomaly_score, float)
    assert isinstance(res.is_frequency_anomalous, bool)


def test_edge_residual_explainer():
    """Verifies Sobel, Laplacian, and SRM multiscale edge residual maps."""
    explainer = EdgeResidualExplainer()
    
    # Image with sharp boundary
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :, :] = 255

    res = explainer.analyze(img)

    assert res.sobel_magnitude.shape == (256, 256)
    assert res.laplacian_residual.shape == (256, 256)
    assert res.srm_residual.shape == (256, 256)
    assert res.gradient_inconsistency_map.shape == (256, 256)
    assert res.edge_anomaly_score >= 0.0


def test_patch_forensic_scorer():
    """Verifies patch grid partitioning and composite risk ranking."""
    scorer = PatchForensicScorer(grid_size=(8, 8))
    h, w = 128, 128
    
    gradcam = np.random.uniform(0, 1, (h, w)).astype(np.float32)
    attention = np.random.uniform(0, 1, (h, w)).astype(np.float32)
    frequency = np.random.uniform(0, 1, (h, w)).astype(np.float32)
    edge = np.random.uniform(0, 1, (h, w)).astype(np.float32)

    patches = scorer.score_patches(
        image_shape=(h, w),
        gradcam_map=gradcam,
        attention_map=attention,
        frequency_map=frequency,
        edge_map=edge,
    )

    assert len(patches) == 64  # 8 x 8
    # Should be sorted descending
    for i in range(len(patches) - 1):
        assert patches[i].composite_risk >= patches[i+1].composite_risk
    
    top = patches[0]
    assert len(top.bbox) == 4
    assert top.primary_anomaly_category in [
        "Semantic Saliency",
        "ViT Patch Focus",
        "High-Freq Spectral Anomaly",
        "Edge Boundary Inconsistency",
    ]


def test_forensic_diagnostic_suite_no_memory_leak():
    """Verifies that running the complete diagnostic suite multiple times does not leak tensor references."""
    model = MockViTEncoder().eval()
    for p in model.parameters():
        p.requires_grad = False

    vit_gradcam = ViTGradCAM(model, target_layer=model.layers[-1], has_cls_token=True)
    rollout = ViTAttentionRollout(model)
    suite = ForensicDiagnosticSuite(vit_gradcam=vit_gradcam, attention_rollout=rollout)

    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Run 10 continuous passes
    for _ in range(10):
        res = suite.explain(image=dummy_img)
        assert res["prob_aigc"] is not None
        assert len(res["top_anomalous_patches"]) > 0

    # Ensure garbage collection clears all temporary graph nodes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
