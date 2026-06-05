"""8-channel ResNet-18 with a scalar regression head (Yeh 2020 protocol).

Random-init (from scratch). The stem is rebuilt for 8 input channels
(RGB + NIR + SWIR1 + SWIR2 + thermal + nightlights). A Dropout layer sits
before the final linear layer so the same trained weights serve MC-dropout
uncertainty later (keep dropout active at inference, sample N forward passes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class PovertyResNet(nn.Module):
    def __init__(self, in_channels: int = 8, dropout: float = 0.2):
        super().__init__()
        net = resnet18(weights=None)  # random init, from scratch
        # rebuild the 7x7 stem for `in_channels` (default 8) instead of 3
        net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                              padding=3, bias=False)
        n_feat = net.fc.in_features  # 512
        net.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(n_feat, 1))
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B, 1) -> (B,)
