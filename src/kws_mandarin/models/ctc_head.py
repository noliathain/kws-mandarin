"""Linear CTC head: encoder features (B, T, D) -> per-frame unit logits (B, T, V)."""

from __future__ import annotations

from torch import Tensor, nn


class CTCHead(nn.Module):
    def __init__(self, enc_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(enc_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)
