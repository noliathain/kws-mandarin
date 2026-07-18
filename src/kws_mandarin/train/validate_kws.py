"""Validation metrics: CTC token-error-rate (fast proxy) and FRR@FAH (the real KWS metric).

Runs streaming over the dev set — model inference is batched, but log-probs are consumed
immediately (greedy TER + keyword scoring) rather than cached, so memory stays flat.

FRR@FAH trial construction: for each validation keyword, a dev utterance is a *positive* if
the keyword Han string is a substring of its transcript, else a *negative*. Every
(keyword, utterance) pair is scored by the CTC keyword spotter, and all trials are pooled
into one DET curve via the eval harness.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..data.manifest import Utterance
from ..decode import CTCKeywordSpotter, NTCKeywordSpotter
from ..eval import summary
from ..tokenizer import PinyinTokenizer


def greedy_decode(log_probs: Tensor, blank: int = 0) -> list[int]:
    """Argmax path -> collapse repeats -> drop blanks, for one (T, V) tensor."""
    ids = log_probs.argmax(dim=-1).tolist()
    out: list[int] = []
    prev = None
    for i in ids:
        if i != prev and i != blank:
            out.append(i)
        prev = i
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _load_wav(path: str, sample_rate: int) -> Tensor:
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


@torch.no_grad()
def run_validation(
    model,
    utts: list[Utterance],
    tokenizer: PinyinTokenizer,
    keywords: list[str],
    device,
    sample_rate: int = 16000,
    batch_size: int = 32,
    max_utts: int | None = None,
    target_fahs: tuple[float, ...] = (0.5, 1.0),
    use_ntc: bool = False,
    ntc_lambda: float = 2.0,
) -> dict:
    model.eval()
    blank = tokenizer.blank_id
    spotter = (
        NTCKeywordSpotter(blank=blank, lambda_ins=ntc_lambda, lambda_mask=ntc_lambda)
        if use_ntc else CTCKeywordSpotter(blank=blank)
    )
    kw_ids = {kw: tokenizer.encode(kw) for kw in keywords}

    if max_utts is not None:
        utts = utts[:max_utts]

    tot_edits = tot_len = 0
    trials: list[tuple[float, int, float]] = []  # (score, label, duration)

    for start in range(0, len(utts), batch_size):
        chunk = utts[start : start + batch_size]
        wavs = [_load_wav(u.wav, sample_rate) for u in chunk]
        lengths = torch.tensor([w.numel() for w in wavs])
        n_max = int(lengths.max())
        batch = torch.zeros(len(wavs), n_max, device=device)
        for i, w in enumerate(wavs):
            batch[i, : w.numel()] = w.to(device)

        logits = model(batch)                        # (B, T, V)
        log_probs = logits.log_softmax(-1)
        out_lens = model.output_lengths(lengths).tolist()

        for i, u in enumerate(chunk):
            lp = log_probs[i, : out_lens[i]].cpu()    # (T_i, V)
            # --- TER ---
            hyp = greedy_decode(lp, blank)
            ref = tokenizer.encode(u.text)
            tot_edits += edit_distance(hyp, ref)
            tot_len += max(1, len(ref))
            # --- keyword trials ---
            for kw, ids in kw_ids.items():
                if not ids:
                    continue
                score = spotter.score(lp, ids)
                label = 1 if kw in u.text else 0
                trials.append((score, label, u.duration))

    metrics = {"ter": tot_edits / max(1, tot_len)}
    if trials and any(t[1] == 1 for t in trials) and any(t[1] == 0 for t in trials):
        scores = [t[0] for t in trials]
        labels = [t[1] for t in trials]
        durations = [t[2] for t in trials]
        for fah, op in summary(scores, labels, durations, target_fahs).items():
            metrics[f"frr@{fah}"] = op.frr
    return metrics
