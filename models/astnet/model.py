"""
models/astnet/model.py
Attention-based residual autoencoder (ASTNet-style).
Inspired by https://github.com/vt-le/astnet (MIT License).
"""
import torch
import torch.nn as nn


class SpatialAttention2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        a = self.op(x)
        return x * a, a


class WideResBlock(nn.Module):
    def __init__(self, c: int, widen: int = 2):
        super().__init__()
        hid = c * widen
        self.net = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, hid, 3, 1, 1),
            nn.BatchNorm2d(hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, c, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class ASTNet(nn.Module):
    """
    Spatial attention residual autoencoder.
    forward(x) -> (reconstruction, attention_map) where attention_map is (B,1,H,W).
    """

    def __init__(self, in_ch: int = 1, base: int = 48):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, 1, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(c1, c2, 4, 2, 1)
        self.enc1 = nn.Sequential(WideResBlock(c2), WideResBlock(c2))
        self.down2 = nn.Conv2d(c2, c3, 4, 2, 1)
        self.enc2 = nn.Sequential(WideResBlock(c3), WideResBlock(c3))
        self.down3 = nn.Conv2d(c3, c4, 4, 2, 1)
        self.enc3 = nn.Sequential(WideResBlock(c4), WideResBlock(c4))
        self.attn = SpatialAttention2d(c4)
        self.enc4 = nn.Sequential(WideResBlock(c4), WideResBlock(c4))

        self.up3 = nn.ConvTranspose2d(c4, c3, 4, 2, 1)
        self.dec3 = nn.Sequential(WideResBlock(c3), WideResBlock(c3))
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, 2, 1)
        self.dec2 = nn.Sequential(WideResBlock(c2), WideResBlock(c2))
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, 2, 1)
        self.dec1 = nn.Sequential(WideResBlock(c1), WideResBlock(c1))
        self.out = nn.Sequential(
            nn.Conv2d(c1, in_ch, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(self.down1(x0))
        x2 = self.enc2(self.down2(x1))
        x3 = self.enc3(self.down3(x2))
        xa, attn = self.attn(x3)
        xb = self.enc4(xa)
        y2 = self.dec3(self.up3(xb) + x2)
        y1 = self.dec2(self.up2(y2) + x1)
        y0 = self.dec1(self.up1(y1) + x0)
        recon = self.out(y0)
        attn_up = nn.functional.interpolate(
            attn, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return recon, attn_up

    def anomaly_score(self, x):
        recon, attn = self.forward(x)
        err = (x - recon) ** 2
        w = attn + 0.25
        return torch.mean(err * w, dim=[1, 2, 3])
