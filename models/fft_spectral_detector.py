#!/usr/bin/env python3
"""2D Fast Fourier Transform (FFT) Frequency-Domain Classifier & Feature Extractor.
Analyzes 2D power spectrum magnitude, azimuthal frequency decay, and upsampling grid peaks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FFTSpectralFeatureExtractor(nn.Module):
    """Computes 2D FFT magnitude spectrum and azimuthal radial frequency profiles.
    
    Generative models (GANs and Diffusion Latent Decoders) leave distinctive periodic
    high-frequency peaks and abnormal spectral power fall-offs in the frequency domain.
    """

    def __init__(self, num_radial_bins=64):
        super().__init__()
        self.num_radial_bins = num_radial_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  B x 3 x H x W (RGB images)
        Output: B x (3 * num_radial_bins + 3 * 3) Spectral Feature Vector
        """
        b, c, h, w = x.shape
        # Compute 2D FFT per channel
        # Center the frequencies with fftshift
        fft = torch.fft.fft2(x, norm="ortho")
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        
        # Log Power Spectrum Magnitude: log(1 + |F(u, v)|)
        magnitude = torch.log1p(torch.abs(fft_shift))
        
        # 1. Radial/Azimuthal Frequency Energy Distribution
        # Create radial coordinate grid
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=x.device) - cy
        x_c = torch.arange(w, device=x.device) - cx
        yy, xx = torch.meshgrid(y, x_c, indexing="ij")
        radius = torch.sqrt(yy**2 + xx**2)
        max_r = float(min(cy, cx))
        
        radial_bins = torch.linspace(0, max_r, self.num_radial_bins + 1, device=x.device)
        radial_profiles = []
        
        for i in range(self.num_radial_bins):
            r_min = radial_bins[i]
            r_max = radial_bins[i + 1]
            mask = (radius >= r_min) & (radius < r_max)
            if mask.sum() > 0:
                # Average magnitude in this radial frequency shell per image and channel
                shell_mean = (magnitude * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1)) / mask.sum()
            else:
                shell_mean = torch.zeros(b, c, device=x.device)
            radial_profiles.append(shell_mean)
            
        radial_feats = torch.stack(radial_profiles, dim=-1) # B x 3 x num_radial_bins
        radial_feats = radial_feats.view(b, -1) # Flatten to B x (3 * num_radial_bins)
        
        # 2. Band Energy Ratios (Low / Mid / High frequency power ratios)
        low_mask = (radius < max_r * 0.33)
        mid_mask = (radius >= max_r * 0.33) & (radius < max_r * 0.66)
        high_mask = (radius >= max_r * 0.66)
        
        low_energy = (magnitude * low_mask).sum(dim=(-2, -1)) / max(low_mask.sum(), 1)
        mid_energy = (magnitude * mid_mask).sum(dim=(-2, -1)) / max(mid_mask.sum(), 1)
        high_energy = (magnitude * high_mask).sum(dim=(-2, -1)) / max(high_mask.sum(), 1)
        
        band_feats = torch.cat([low_energy, mid_energy, high_energy], dim=-1) # B x 9
        
        # Combined Spectral Descriptor
        spectral_feats = torch.cat([radial_feats, band_feats], dim=-1)
        return spectral_feats


class FFTEnergyClassifierHead(nn.Module):
    """Classifier head operating directly on FFT spectral descriptors."""
    def __init__(self, in_features=3*64 + 9, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, spectral_feats: torch.Tensor) -> torch.Tensor:
        return self.net(spectral_feats)
