"""
HSTforU-style next-frame predictor: PVTv2 encoder + temporal HST blocks + U-Net decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hstforu.hst_module import HSTBlock
from models.hstforu.pvt_v2 import build_pvt_encoder


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.merge = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.merge(torch.cat([x, skip], dim=1))


class HSTforU(nn.Module):
    """
    Args:
        encoder_name: timm PVTv2 variant (default pvt_v2_b2).
        clip_length: total frames in training clip; model sees clip_length - 1 inputs.
        img_size: unused (API compatibility with PRD).
    Forward:
        x: (B, T, C, H, W) with T == clip_length - 1
        returns predicted next frame (B, C, H, W)
    """

    def __init__(
        self,
        encoder_name: str = "pvt_v2_b2",
        clip_length: int = 5,
        img_size: int = 256,
        in_channels: int = 1,
    ):
        super().__init__()
        self.clip_length = clip_length
        self.T = clip_length - 1
        self.encoder = build_pvt_encoder(in_channels, encoder_name)
        chs = list(self.encoder.feature_info.channels())
        self.hst = nn.ModuleList([HSTBlock(c) for c in chs])

        c0, c1, c2, c3 = chs
        self.dec2 = DecoderBlock(c3, c2, c2)
        self.dec1 = DecoderBlock(c2, c1, c1)
        self.dec0 = DecoderBlock(c1, c0, c0)
        self.head = nn.Sequential(
            nn.ConvTranspose2d(c0, c0 // 2, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(c0 // 2),
            nn.GELU(),
            nn.Conv2d(c0 // 2, in_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        if t != self.T:
            raise ValueError(f"Expected T={self.T}, got {t}")

        per_level: list[list[torch.Tensor]] = [[] for _ in range(4)]
        for i in range(t):
            feats = self.encoder(x[:, i])
            for j, fj in enumerate(feats):
                per_level[j].append(fj)

        fused = []
        for j in range(4):
            s = torch.stack(per_level[j], dim=2)
            fused.append(self.hst[j](s))

        f0, f1, f2, f3 = fused
        y = self.dec2(f3, f2)
        y = self.dec1(y, f1)
        y = self.dec0(y, f0)
        out = self.head(y)
        if out.shape[-2:] != (h, w):
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out
