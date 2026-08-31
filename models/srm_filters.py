import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMConvolution(nn.Module):
    """A lightweight fixed high-pass residual filter."""

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 0.25, 0.0],
             [0.25, -1.0, 0.25],
             [0.0, 0.25, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=1, groups=3)


class WaveletResidualBlock(nn.Module):
    """SRM residual extraction followed by a 2-D Haar detail decomposition.

    Input:  B x 3 x H x W
    Output: B x 9 x H/2 x W/2 (LH, HL, HH for each RGB channel)
    """

    def __init__(self):
        super().__init__()
        self.srm = SRMConvolution()

    @staticmethod
    def haar_detail_bands(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        h -= h % 2
        w -= w % 2
        x = x[..., :h, :w]

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        # Detail bands only; approximation is deliberately omitted.
        lh = (-x00 + x01 - x10 + x11) / 2.0
        hl = (-x00 - x01 + x10 + x11) / 2.0
        hh = (x00 - x01 - x10 + x11) / 2.0
        return torch.cat([lh, hl, hh], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.haar_detail_bands(self.srm(x))
