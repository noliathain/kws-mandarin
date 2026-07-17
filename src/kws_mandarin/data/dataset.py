"""Torch Dataset + collate for CTC training from manifests.

The manifest stores raw Han text; tokenization to unit ids happens here, on the fly, so the
unit mode is a training-time choice (never a re-prep). Waveforms load lazily and are
downmixed to mono / resampled to the model rate. An optional ``augment`` callable (MUSAN/RIR)
is applied per item at train time.

Padded batch frames are harmless: we pass true per-utterance ``wav_lengths`` so the trainer
derives real input lengths and CTC masks the padding.
"""

from __future__ import annotations

from collections.abc import Callable

import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from ..tokenizer import PinyinTokenizer
from .manifest import Utterance, read_manifest


class KWSDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        tokenizer: PinyinTokenizer,
        sample_rate: int = 16000,
        augment: Callable[[Tensor], Tensor] | None = None,
        min_units: int = 1,
    ):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.augment = augment
        utts = read_manifest(manifest_path)
        # Drop utterances that tokenize to fewer than min_units (CTC needs a non-empty target).
        self.utts: list[Utterance] = [u for u in utts if len(tokenizer.encode(u.text)) >= min_units]

    def __len__(self) -> int:
        return len(self.utts)

    def _load_wav(self, path: str) -> Tensor:
        # soundfile decodes standard PCM wav without the torchcodec backend torchaudio.load
        # now requires; torchaudio is used only for pure-tensor resampling.
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (N, C)
        wav = torch.from_numpy(data).mean(dim=1)                    # mono (N,)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def __getitem__(self, idx: int) -> dict:
        u = self.utts[idx]
        wav = self._load_wav(u.wav)
        if self.augment is not None:
            wav = self.augment(wav)
        target = torch.tensor(self.tokenizer.encode(u.text), dtype=torch.long)
        return {"wav": wav, "target": target, "utt_id": u.utt_id}


def collate_kws(batch: list[dict]) -> dict:
    """Pad waveforms; concatenate CTC targets. Returns tensors ready for KWSModel + CTCLoss."""
    wav_lengths = torch.tensor([b["wav"].numel() for b in batch], dtype=torch.long)
    n_max = int(wav_lengths.max())
    wavs = torch.zeros(len(batch), n_max)
    for i, b in enumerate(batch):
        wavs[i, : b["wav"].numel()] = b["wav"]

    targets = torch.cat([b["target"] for b in batch])
    target_lengths = torch.tensor([b["target"].numel() for b in batch], dtype=torch.long)
    return {
        "wavs": wavs,
        "wav_lengths": wav_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "utt_ids": [b["utt_id"] for b in batch],
    }
