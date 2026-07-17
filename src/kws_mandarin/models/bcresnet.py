"""BC-ResNet encoder (Kim et al., 2021) — broadcasted residual learning.

Each block factors 2-D convolution into two cheap paths that are summed (the "broadcasted
residual"):

* **frequency path** ``f2``: a depthwise conv over frequency + SubSpectralNorm. Keeps the
  full (F, T) map.
* **temporal path** ``f1``: average the freq-path output over frequency, run a depthwise
  conv over *time*, a pointwise mix, then broadcast back across frequency and add.

This gives most of a full 2-D conv's modeling power at a fraction of the parameters, which
is why BC-ResNet is the right low-param KWS backbone.

**Streaming (D-latency).** With ``causal=True`` every temporal convolution is left-padded
only (no lookahead), and time stride is 1 throughout, so the CTC frame rate stays at 10 ms
and output frame ``t`` depends only on input frames ``<= t``. Frequency is subsampled across
stages; time is not. The exact channel schedule here is a sensible scaled default — reconcile
widths with the official reference before locking a shipping config.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .subspectralnorm import SubSpectralNorm


def _round(x: float) -> int:
    return max(1, int(round(x)))


class _CausalConv2d(nn.Module):
    """Conv2d that is causal in time (left-pad only) and 'same' in frequency.

    ``freq_stride`` subsamples frequency; time stride is always 1.
    """

    def __init__(self, in_c: int, out_c: int, freq_k: int, time_k: int,
                 freq_stride: int = 1, groups: int = 1, causal: bool = True):
        super().__init__()
        self.time_k = time_k
        self.freq_pad = freq_k // 2
        self.causal = causal
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=(freq_k, time_k),
            stride=(freq_stride, 1), padding=0, groups=groups, bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:  # (B, C, F, T)
        if self.causal:
            time_pad = (self.time_k - 1, 0)          # left only
        else:
            time_pad = ((self.time_k - 1) // 2, self.time_k // 2)
        # F.pad order is (last dim left/right, then next): (T_l, T_r, F_l, F_r)
        x = F.pad(x, (*time_pad, self.freq_pad, self.freq_pad))
        return self.conv(x)


class BCResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, freq_stride: int = 1, ssn_bands: int = 5,
                 freq_k: int = 3, time_k: int = 3, causal: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_identity = in_c == out_c and freq_stride == 1
        self.pw_in = None if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)

        # frequency path (depthwise over freq, time kernel 1 -> never touches time)
        self.f2 = _CausalConv2d(out_c, out_c, freq_k=freq_k, time_k=1,
                                freq_stride=freq_stride, groups=out_c, causal=causal)
        self.ssn = SubSpectralNorm(out_c, ssn_bands)

        # temporal path (on freq-averaged features): depthwise-time -> BN -> ReLU -> pointwise
        self.time_k = time_k
        self.causal = causal
        self.f1_dw = nn.Conv1d(out_c, out_c, time_k, groups=out_c, bias=False)
        self.f1_bn = nn.BatchNorm1d(out_c)
        self.f1_pw = nn.Conv1d(out_c, out_c, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:  # (B, C_in, F, T)
        identity = x
        if self.pw_in is not None:
            x = self.pw_in(x)                        # (B, C_out, F, T)

        f2 = self.ssn(self.f2(x))                    # (B, C_out, F', T)

        a = f2.mean(dim=2)                           # broadcast over freq -> (B, C_out, T)
        pad = (self.time_k - 1, 0) if self.causal else ((self.time_k - 1) // 2, self.time_k // 2)
        a = self.f1_dw(F.pad(a, pad))
        a = self.act(self.f1_bn(a))
        a = self.drop(self.f1_pw(a))
        a = a.unsqueeze(2)                           # (B, C_out, 1, T)

        out = f2 + a                                 # broadcasted residual
        if self.use_identity:
            out = out + identity
        return self.act(out)


class BCResNetEncoder(nn.Module):
    """Log-mel (B, n_mels, T) -> encoded sequence (B, T, enc_dim). Time rate preserved."""

    def __init__(self, n_mels: int = 40, scale: float = 3.0, ssn_bands: int = 5,
                 causal: bool = True, dropout: float = 0.1,
                 n_blocks: tuple[int, ...] = (2, 2, 4, 4),
                 freq_strides: tuple[int, ...] = (1, 2, 2, 1),
                 width_mult: tuple[float, ...] = (8.0, 12.0, 16.0, 20.0)):
        super().__init__()
        assert len(n_blocks) == len(freq_strides) == len(width_mult)
        channels = [_round(scale * w) for w in width_mult]
        self.enc_dim = channels[-1]
        self.causal = causal

        # stem: subsample frequency by 2, causal in time
        self.stem = nn.Sequential(
            _CausalConv2d(1, channels[0], freq_k=5, time_k=5, freq_stride=2, causal=causal),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
        )

        blocks: list[nn.Module] = []
        in_c = channels[0]
        for out_c, nb, fs in zip(channels, n_blocks, freq_strides):
            for i in range(nb):
                blocks.append(BCResBlock(
                    in_c, out_c, freq_stride=(fs if i == 0 else 1),
                    ssn_bands=ssn_bands, causal=causal, dropout=dropout,
                ))
                in_c = out_c
        self.blocks = nn.Sequential(*blocks)

        # head: depthwise over remaining freq, then collapse frequency
        self.head = _CausalConv2d(in_c, in_c, freq_k=5, time_k=1, groups=in_c, causal=causal)

    @property
    def subsampling_factor(self) -> int:
        return 1  # time resolution preserved (10 ms/frame)

    def output_lengths(self, input_lengths: Tensor) -> Tensor:
        return input_lengths  # no temporal subsampling

    def forward(self, feats: Tensor) -> Tensor:
        """``feats`` (B, n_mels, T) -> (B, T, enc_dim)."""
        x = feats.unsqueeze(1)          # (B, 1, F, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)                # (B, C, F', T)
        x = x.mean(dim=2)               # collapse frequency -> (B, C, T)
        return x.transpose(1, 2).contiguous()  # (B, T, C)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
