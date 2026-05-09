"""
Hierarchical spatio-temporal mixing along a short frame stack.
"""
import torch
import torch.nn as nn


class HSTBlock(nn.Module):
    """
    Input:  (B, C, T, H, W)
    Output: (B, C, H, W) — uses last temporal slice after 3D mixing + residual.
    """

    def __init__(self, channels: int, temporal_kernel: int = 3):
        super().__init__()
        pad = temporal_kernel // 2
        groups = min(32, channels)
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(pad, 0, 0),
            groups=groups,
        )
        self.pw = nn.Conv3d(channels, channels, 1)
        self.bn = nn.BatchNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.pw(self.conv(x)) + x))
        return out[:, :, -1, :, :]
