"""
PVTv2 backbone wrapper (timm) for multi-scale features.
"""
from __future__ import annotations

import torch.nn as nn


def build_pvt_encoder(in_chans: int = 1, model_name: str = "pvt_v2_b2") -> nn.Module:
    import timm

    return timm.create_model(
        model_name,
        pretrained=False,
        in_chans=in_chans,
        features_only=True,
        out_indices=(0, 1, 2, 3),
    )
