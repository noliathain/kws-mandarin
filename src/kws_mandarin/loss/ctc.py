"""CTC training loss.

A thin wrapper over ``torch.nn.functional.ctc_loss`` that owns the two conventions the rest
of the codebase relies on: **blank is id 0** (matches the tokenizer) and logits arrive as
(B, T, V). It handles the log-softmax and the (T, B, V) transpose CTC wants.

This is the plain-CTC objective. The NTC-style *wildcard-arc* variant (self-loop = noise
insertion, bypass = masking) will subclass/extend this — the highest-leverage robustness
change (D2) — and slots in here without touching the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CTCLoss(nn.Module):
    def __init__(self, blank: int = 0, zero_infinity: bool = True, reduction: str = "mean"):
        super().__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,          # (B, T, V)
        targets: Tensor,         # (sum(target_lengths),) or (B, S)
        input_lengths: Tensor,   # (B,)
        target_lengths: Tensor,  # (B,)
    ) -> Tensor:
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
        return F.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            zero_infinity=self.zero_infinity,
            reduction=self.reduction,
        )
