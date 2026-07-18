"""Open-vocabulary keyword search over CTC posteriors.

A keyword is a runtime unit-id sequence (from the tokenizer). We score how strongly it is
present in a frame of CTC posteriors. Two primitives:

* ``ctc_forward_logprob`` — exact CTC forward log-likelihood that the *whole* posterior
  spells the target. Matches ``-F.ctc_loss`` (its correctness anchor).
* ``ctc_keyword_score`` — the spotting score: an any-start / any-end CTC forward
  (keyword-filler model). Non-keyword frames are absorbed for free, the keyword may begin
  and end at any frame, and the score is the best-scoring occurrence. This is a monotone
  detection score suitable for thresholding against the FRR@FAH harness.

``CTCKeywordSpotter`` wraps the spotting score behind a stable interface so the MFA-KWS
Token-and-Duration Transducer decoder can replace it later without touching callers.

Design note: NTC-style *wildcard arcs* (self-loop = noise insertion, bypass = masking) are a
modification of exactly this forward recursion — the next robustness step (D2/D7).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor

_NEG = float("-inf")


def _extended(keyword_ids: Sequence[int], blank: int, device) -> tuple[Tensor, Tensor]:
    """Blank-interleaved state sequence and the skip-transition mask for CTC."""
    ids = list(keyword_ids)
    if len(ids) == 0:
        raise ValueError("keyword must have at least one unit")
    ext: list[int] = [blank]
    for u in ids:
        ext.extend((u, blank))
    ext_t = torch.tensor(ext, dtype=torch.long, device=device)  # (S,), S = 2U+1
    s = len(ext)
    skip = torch.zeros(s, dtype=torch.bool, device=device)
    for i in range(2, s):
        # a skip s-2 -> s is allowed only into a label state distinct from the last label
        if ext[i] != blank and ext[i] != ext[i - 2]:
            skip[i] = True
    return ext_t, skip


def _transition(prev: Tensor, skip: Tensor) -> Tensor:
    """CTC forward transition: stay, advance from s-1, or skip from s-2 (log domain)."""
    neg1 = prev.new_full((1,), _NEG)
    neg2 = prev.new_full((2,), _NEG)
    term1 = torch.cat([neg1, prev[:-1]])                       # from s-1
    term2 = torch.where(skip, torch.cat([neg2, prev[:-2]]), prev.new_full(prev.shape, _NEG))
    return torch.logaddexp(torch.logaddexp(prev, term1), term2)


def ctc_forward_logprob(log_probs: Tensor, keyword_ids: Sequence[int], blank: int = 0) -> Tensor:
    """Exact CTC forward log-likelihood that (T, V) log-probs spell ``keyword_ids`` in full."""
    if log_probs.dim() != 2:
        raise ValueError(f"expected (T, V) log-probs, got {tuple(log_probs.shape)}")
    ext, skip = _extended(keyword_ids, blank, log_probs.device)
    s = ext.numel()
    t_len = log_probs.shape[0]

    alpha = log_probs.new_full((s,), _NEG)
    emit0 = log_probs[0].index_select(0, ext)
    alpha[0] = emit0[0]
    if s > 1:
        alpha[1] = emit0[1]
    for t in range(1, t_len):
        alpha = _transition(alpha, skip) + log_probs[t].index_select(0, ext)
    return torch.logaddexp(alpha[-1], alpha[-2]) if s >= 2 else alpha[-1]


def ctc_keyword_score(
    log_probs: Tensor,
    keyword_ids: Sequence[int],
    blank: int = 0,
    normalize: bool = True,
) -> Tensor:
    """Any-start / any-end CTC keyword-filler detection score for (T, V) log-probs.

    Returns the best-scoring occurrence (per-unit normalized if ``normalize``). Higher means
    a stronger match; the value is a monotone detection score, not a probability.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"expected (T, V) log-probs, got {tuple(log_probs.shape)}")
    ext, skip = _extended(keyword_ids, blank, log_probs.device)
    s = ext.numel()
    u = len(keyword_ids)
    t_len = log_probs.shape[0]

    # fresh-start boost: the keyword may begin (leading blank s=0 or first label s=1) at any t
    start_boost = log_probs.new_full((s,), _NEG)
    start_boost[0] = 0.0
    if s > 1:
        start_boost[1] = 0.0

    alpha = log_probs.new_full((s,), _NEG)  # "previous" before the first frame
    best = log_probs.new_full((), _NEG)
    for t in range(t_len):
        trans = _transition(alpha, skip) if t > 0 else log_probs.new_full((s,), _NEG)
        alpha = torch.logaddexp(trans, start_boost) + log_probs[t].index_select(0, ext)
        end = torch.logaddexp(alpha[-1], alpha[-2])  # keyword may end at this frame
        best = torch.maximum(best, end)
    return best / u if normalize else best


def _wild_logprob(log_probs: Tensor, blank: int) -> Tensor:
    """Per-frame wildcard/noise score = log of the non-blank probability mass = log(1 - p_blank).

    High when a frame carries token-like content (speech or noise) rather than silence — the
    frames where a low-SNR model tends to emit spurious tokens. This is what the NTC wildcard
    arcs absorb.
    """
    p_blank = log_probs[:, blank].exp().clamp(max=1.0 - 1e-6)
    return torch.log1p(-p_blank).clamp(min=-30.0)


def ntc_keyword_score(
    log_probs: Tensor,
    keyword_ids: Sequence[int],
    blank: int = 0,
    lambda_ins: float = 2.0,
    lambda_mask: float = 2.0,
    normalize: bool = True,
) -> Tensor:
    """NTC-style noise-aware keyword score: CTC keyword-filler + wildcard arcs.

    Realizes NTC-KWS's two robustness arcs directly in the CTC forward recursion (no WFST/k2,
    per D7), by making a *wildcard/noise* emission available at every state at a penalty:

    * **self-loop noise arc** (insertion): at a blank state the frame may be emitted as blank
      *or* as noise (``_wild_logprob`` − ``lambda_ins``), so spurious tokens inserted between
      keyword units are absorbed instead of tanking the alignment.
    * **bypass arc** (masking): at a keyword-token state the frame may be the token *or* noise
      (``_wild_logprob`` − ``lambda_mask``), so a unit masked by noise can still be traversed.

    As ``lambda_* → ∞`` the wildcard vanishes and this reduces exactly to ``ctc_keyword_score``.
    Larger penalties = less noise tolerance. Returns a monotone detection score.
    """
    if log_probs.dim() != 2:
        raise ValueError(f"expected (T, V) log-probs, got {tuple(log_probs.shape)}")
    ext, skip = _extended(keyword_ids, blank, log_probs.device)
    s = ext.numel()
    u = len(keyword_ids)
    t_len = log_probs.shape[0]
    is_blank = ext.eq(blank)
    wild = _wild_logprob(log_probs, blank)  # (T,)

    start_boost = log_probs.new_full((s,), _NEG)
    start_boost[0] = 0.0
    if s > 1:
        start_boost[1] = 0.0

    alpha = log_probs.new_full((s,), _NEG)
    best = log_probs.new_full((), _NEG)
    for t in range(t_len):
        base = log_probs[t].index_select(0, ext)                      # (S,) real emission
        pen = torch.where(is_blank, wild[t] - lambda_ins, wild[t] - lambda_mask)
        emit = torch.logaddexp(base, pen)                             # token/blank OR wildcard
        trans = _transition(alpha, skip) if t > 0 else log_probs.new_full((s,), _NEG)
        alpha = torch.logaddexp(trans, start_boost) + emit
        end = torch.logaddexp(alpha[-1], alpha[-2])
        best = torch.maximum(best, end)
    return best / u if normalize else best


class BaseKeywordSpotter(ABC):
    """Stable open-vocab decoder interface. A TDT decoder implements the same ``score``."""

    @abstractmethod
    def score(self, log_probs: Tensor, keyword_ids: Sequence[int]) -> float:
        ...


class CTCKeywordSpotter(BaseKeywordSpotter):
    def __init__(self, blank: int = 0, normalize: bool = True):
        self.blank = blank
        self.normalize = normalize

    def score(self, log_probs: Tensor, keyword_ids: Sequence[int]) -> float:
        with torch.no_grad():
            return float(ctc_keyword_score(log_probs, keyword_ids, self.blank, self.normalize))


class NTCKeywordSpotter(BaseKeywordSpotter):
    """Noise-aware NTC keyword spotter (wildcard arcs). Drop-in for CTCKeywordSpotter."""

    def __init__(self, blank: int = 0, lambda_ins: float = 2.0, lambda_mask: float = 2.0,
                 normalize: bool = True):
        self.blank = blank
        self.lambda_ins = lambda_ins
        self.lambda_mask = lambda_mask
        self.normalize = normalize

    def score(self, log_probs: Tensor, keyword_ids: Sequence[int]) -> float:
        with torch.no_grad():
            return float(ntc_keyword_score(
                log_probs, keyword_ids, self.blank, self.lambda_ins, self.lambda_mask, self.normalize
            ))
