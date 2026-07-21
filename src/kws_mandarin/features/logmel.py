"""Log-mel front-end.

40-bin log-mel at 25 ms / 10 ms is the standard KWS input and what BC-ResNet expects.
Kept as an ``nn.Module`` (not a dataloader transform) so it travels with the model into
export — the same code computes features at train and inference time, and there is no
train/serve feature skew.

Frame count with ``center=True`` (the torchaudio default): ``T = 1 + num_samples // hop``.
At 16 kHz with hop 160 that is one frame per 10 ms plus one, e.g. 1 s -> 101 frames.
"""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor, nn


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,          # 25 ms
        hop_length: int = 160,     # 10 ms
        win_length: int = 400,     # 25 ms
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        log_eps: float = 1e-6,
        per_utt_norm: bool = True,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.log_eps = log_eps
        self.per_utt_norm = per_utt_norm
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
        )

    @property
    def output_dim(self) -> int:
        return self.n_mels

    def num_frames(self, num_samples: int) -> int:
        return 1 + num_samples // self.hop_length

    def forward(self, wav: Tensor, lengths: Tensor | None = None) -> Tensor:
        """``wav`` (B, N) or (B, 1, N) float in [-1, 1] -> log-mel (B, n_mels, T).

        ``lengths`` (B,) = valid sample counts. When given, CMVN statistics are computed over
        *valid frames only* and padding is zeroed — otherwise a 4 s clip padded to 14 s has its
        mean/std dominated by silence, distorting the real speech.
        """
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        if wav.dim() != 2:
            raise ValueError(f"expected (B, N) or (B, 1, N), got shape {tuple(wav.shape)}")
        mel = self.mel(wav)                       # (B, n_mels, T)
        feat = torch.log(mel + self.log_eps)
        if not self.per_utt_norm:
            return feat
        if lengths is None:
            mean = feat.mean(dim=-1, keepdim=True)
            std = feat.std(dim=-1, keepdim=True).clamp_min(1e-5)
            return (feat - mean) / std
        # masked CMVN: stats over valid frames only
        frames = (1 + lengths.to(feat.device) // self.hop_length).clamp(max=feat.shape[-1])
        m = (torch.arange(feat.shape[-1], device=feat.device)[None, :] < frames[:, None])
        m = m.unsqueeze(1).to(feat.dtype)         # (B, 1, T)
        cnt = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (feat * m).sum(dim=-1, keepdim=True) / cnt
        var = (((feat - mean) * m) ** 2).sum(dim=-1, keepdim=True) / cnt
        feat = (feat - mean) / var.sqrt().clamp_min(1e-5)
        return feat * m                            # zero padding so the encoder sees clean zeros
