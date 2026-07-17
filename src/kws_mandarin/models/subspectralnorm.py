"""SubSpectralNorm (Chang et al., 2021).

Splits the frequency axis into ``sub_bands`` groups and normalizes each group with its own
BatchNorm statistics — different frequency regions carry different acoustic content, so one
global BN is too coarse. This is the normalization used inside BC-ResNet.

Export note (D1): at int8 export the per-sub-band affine folds into the preceding conv.
Implemented as a plain ``BatchNorm2d`` over ``channels * sub_bands`` folded channels, which
is exactly what folds cleanly under PT2E — no custom op.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SubSpectralNorm(nn.Module):
    def __init__(self, channels: int, sub_bands: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        """``x`` (B, C, F, T) -> (B, C, F, T); F must be divisible by ``sub_bands``."""
        b, c, f, t = x.shape
        if f % self.sub_bands != 0:
            raise ValueError(
                f"frequency dim {f} not divisible by sub_bands {self.sub_bands}"
            )
        x = x.reshape(b, c * self.sub_bands, f // self.sub_bands, t)
        x = self.bn(x)
        return x.reshape(b, c, f, t)
