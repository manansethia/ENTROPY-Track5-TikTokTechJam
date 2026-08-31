#!/usr/bin/env python3
"""E²GenF-style Edge and Generative Boundary Artifact Specialist.
Extracts Sobel gradient magnitudes, Laplacian second-order derivatives,
and high-pass edge boundary residuals to catch generative upsampling artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeArtifactFeatureExtractor(nn.Module):
    """Extracts multiscale gradient and edge boundary descriptors from RGB images."""

    def __init__(self, out_dim=256):
        super().__init__()
        # Sobel Horizontal and Vertical Kernels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        # Laplacian 2nd order derivative
        laplacian = torch.tensor(
            [[ 0.0,  1.0,  0.0],
             [ 1.0, -4.0,  1.0],
             [ 0.0,  1.0,  0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1))
        self.register_buffer("laplacian", laplacian.repeat(3, 1, 1, 1))

        # Lightweight CNN backbone operating on edge maps
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1), # 3 sobel_x, 3 sobel_y, 3 laplacian
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  B x 3 x H x W (RGB images)
        Output: B x out_dim edge feature descriptor
        """
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        lap = F.conv2d(x, self.laplacian, padding=1, groups=3)
        
        edge_tensor = torch.cat([gx, gy, lap], dim=1) # B x 9 x H x W
        return self.edge_encoder(edge_tensor)
