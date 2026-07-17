"""End-to-end KWS acoustic model: waveform -> per-frame unit logits.

    wav (B, N) -> LogMelFrontend -> BCResNetEncoder -> CTCHead -> logits (B, T, V)

Frontend lives inside the model on purpose (no train/serve feature skew). Blank is id 0,
consistent with the tokenizer and the CTC loss.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..features import LogMelFrontend
from .bcresnet import BCResNetEncoder
from .ctc_head import CTCHead


class KWSModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 40,
        scale: float = 3.0,
        causal: bool = True,
        ssn_bands: int = 5,
        dropout: float = 0.1,
        blank_id: int = 0,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        self.frontend = LogMelFrontend(n_mels=n_mels)
        self.encoder = BCResNetEncoder(
            n_mels=n_mels, scale=scale, ssn_bands=ssn_bands, causal=causal, dropout=dropout
        )
        self.head = CTCHead(self.encoder.enc_dim, vocab_size)

    def forward(self, wav: Tensor) -> Tensor:
        """``wav`` (B, N) or (B, 1, N) -> logits (B, T, V)."""
        feats = self.frontend(wav)          # (B, n_mels, T)
        enc = self.encoder(feats)           # (B, T, D)
        return self.head(enc)               # (B, T, V)

    def log_probs(self, wav: Tensor) -> Tensor:
        return self.forward(wav).log_softmax(dim=-1)

    def output_lengths(self, input_samples: Tensor) -> Tensor:
        frames = 1 + input_samples // self.frontend.hop_length
        return self.encoder.output_lengths(frames)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
