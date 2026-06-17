"""Monocular depth network: a U-Net with a pretrained ResNet encoder.

ARCHITECTURE
------------
Encoder: a torchvision ResNet (default resnet18) pretrained on ImageNet. It turns
the RGB image into feature maps at decreasing resolution / increasing semantics
(/2, /4, /8, /16, /32).

Decoder: mirror path that upsamples back up, each step concatenating the matching
encoder feature map (a "skip connection") so sharp details survive the squeeze.

Head: a 1-channel conv. We pass it through ``softplus`` so depth is always
positive, then clamp to [min_depth, max_depth] metres. Predicting metric depth
directly (not relative) is what lets us feed it back into the geometry pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    """Upsample x2, concat the skip feature, then two convs."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = _ConvBlock(in_ch + skip_ch, out_ch)
        self.conv2 = _ConvBlock(out_ch, out_ch)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # guard against off-by-one sizes from odd input dims
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


_BACKBONE_CHANNELS = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
}


class DepthResNet(nn.Module):
    """Predict metric depth (metres) from a single RGB image."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        min_depth: float = 0.5,
        max_depth: float = 80.0,
    ):
        super().__init__()
        if backbone not in _BACKBONE_CHANNELS:
            raise ValueError(f"unsupported backbone {backbone}")
        self.min_depth, self.max_depth = min_depth, max_depth

        weights = "DEFAULT" if pretrained else None
        net = getattr(models, backbone)(weights=weights)
        c = _BACKBONE_CHANNELS[backbone]

        # Encoder stages (expose intermediate features for skips).
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)  # /2, c[0]
        self.pool = net.maxpool                                  # /4
        self.layer1, self.layer2 = net.layer1, net.layer2        # /4 c[1], /8 c[2]
        self.layer3, self.layer4 = net.layer3, net.layer4        # /16 c[3], /32 c[4]

        # Decoder (upsample back to /1, concatenating skips).
        self.up4 = _Up(c[4], c[3], 256)   # /32 -> /16
        self.up3 = _Up(256, c[2], 128)    # /16 -> /8
        self.up2 = _Up(128, c[1], 64)     # /8  -> /4
        self.up1 = _Up(64, c[0], 64)      # /4  -> /2
        self.up0 = _Up(64, 0, 32)         # /2  -> /1
        self.head = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        s0 = self.stem(x)          # /2
        s1 = self.layer1(self.pool(s0))  # /4
        s2 = self.layer2(s1)       # /8
        s3 = self.layer3(s2)       # /16
        s4 = self.layer4(s3)       # /32
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = self.up0(d)
        raw = self.head(d)
        depth = F.softplus(raw)    # strictly positive
        return depth.clamp(self.min_depth, self.max_depth)
