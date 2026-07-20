"""Multi-condition augmentation — the data-side robustness pillar (D2).

Two domains:

* **Waveform** (``WaveformAugment``): speed perturb -> RIR reverberation -> additive noise at a
  sampled SNR -> gain. The order is physically motivated: reverberate the clean signal in a
  room, *then* add the room's ambient/babble noise, *then* apply capture gain. Each stage
  fires stochastically so the model sees a spread of conditions (multi-condition training),
  including some clean audio.
* **Feature** (``SpecAugment``): time/frequency masking on the log-mel, applied in the model
  path at train time.

Noise/RIR sources are injected as samplers (callables returning a waveform tensor) so the
augmenter is decoupled from the filesystem and fully testable; ``WaveformAugment.from_dirs``
wires up MUSAN + RIR directories for real training.

Robustness NOT covered here (tracked separately): NTC-style wildcard-arc CTC (decoder-side
noise modeling, D2/D7) and LLM-generated tone-confusable hard negatives (false-accept
suppression). Waveform+feature augmentation is necessary but not sufficient on its own.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch import Tensor, nn

Sampler = Callable[[], Tensor]


# --- primitives -----------------------------------------------------------------------

def _match_length(noise: Tensor, n: int, rng: random.Random) -> Tensor:
    """Tile/crop a noise clip to exactly ``n`` samples, starting at a random offset."""
    m = noise.numel()
    if m == 0:
        return torch.zeros(n)
    if m < n:
        reps = (n + m - 1) // m
        noise = noise.repeat(reps)
        m = noise.numel()
    start = rng.randint(0, m - n) if m > n else 0
    return noise[start : start + n]


def add_noise(wav: Tensor, noise: Tensor, snr_db: float, rng: random.Random) -> Tensor:
    """Add ``noise`` to ``wav`` scaled so the resulting SNR equals ``snr_db``."""
    noise = _match_length(noise, wav.numel(), rng)
    clean_power = wav.pow(2).mean()
    noise_power = noise.pow(2).mean()
    if noise_power <= 0 or clean_power <= 0:
        return wav
    scale = torch.sqrt(clean_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return wav + scale * noise


def apply_rir(wav: Tensor, rir: Tensor) -> Tensor:
    """Convolve with a room impulse response, preserving length and level.

    The RIR is unit-energy normalized (so it does not change loudness), and the output is
    aligned to the RIR's direct-path peak and trimmed back to the input length.
    """
    n = wav.numel()
    energy = rir.pow(2).sum().sqrt()
    if energy <= 0:
        return wav
    rir = rir / energy
    peak = int(torch.argmax(rir.abs()))
    conv = torch.nn.functional.conv1d(
        wav.view(1, 1, -1),
        rir.flip(0).view(1, 1, -1),
        padding=rir.numel() - 1,
    ).view(-1)
    return conv[peak : peak + n]


def speed_perturb(wav: Tensor, factor: float, sample_rate: int) -> Tensor:
    """Speed up (factor>1) / slow down (<1) by resampling. Changes duration and pitch."""
    if factor == 1.0:
        return wav
    new_sr = int(round(sample_rate / factor))
    return torchaudio.functional.resample(wav, sample_rate, new_sr)


def gain_perturb(wav: Tensor, gain_db: float) -> Tensor:
    return wav * (10.0 ** (gain_db / 20.0))


# --- feature-domain -------------------------------------------------------------------

class SpecAugment(nn.Module):
    """Time/frequency masking on log-mel features (B, n_mels, T). Train-time only."""

    def __init__(self, freq_mask: int = 8, n_freq: int = 2,
                 time_mask: int = 25, n_time: int = 2, mask_value: float = 0.0):
        super().__init__()
        self.freq_mask = freq_mask
        self.n_freq = n_freq
        self.time_mask = time_mask
        self.n_time = n_time
        self.mask_value = mask_value

    def forward(self, feat: Tensor) -> Tensor:
        if not self.training:
            return feat
        feat = feat.clone()
        b, n_mels, t = feat.shape
        for i in range(b):
            for _ in range(self.n_freq):
                f = int(torch.randint(0, self.freq_mask + 1, (1,)))
                if f > 0 and n_mels - f > 0:
                    f0 = int(torch.randint(0, n_mels - f, (1,)))
                    feat[i, f0 : f0 + f, :] = self.mask_value
            for _ in range(self.n_time):
                w = int(torch.randint(0, self.time_mask + 1, (1,)))
                if w > 0 and t - w > 0:
                    t0 = int(torch.randint(0, t - w, (1,)))
                    feat[i, :, t0 : t0 + w] = self.mask_value
        return feat


# --- composite waveform augmenter -----------------------------------------------------

class WaveformAugment:
    def __init__(
        self,
        noise_sampler: Sampler | None = None,
        rir_sampler: Sampler | None = None,
        sample_rate: int = 16000,
        snr_db_range: tuple[float, float] = (0.0, 20.0),
        p_noise: float = 0.6,
        p_rir: float = 0.5,
        speed_factors: tuple[float, ...] = (0.9, 1.0, 1.1),
        p_speed: float = 0.5,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        p_gain: float = 0.5,
        seed: int | None = None,
    ):
        self.noise_sampler = noise_sampler
        self.rir_sampler = rir_sampler
        self.sample_rate = sample_rate
        self.snr_db_range = snr_db_range
        self.p_noise = p_noise
        self.p_rir = p_rir
        self.speed_factors = speed_factors
        self.p_speed = p_speed
        self.gain_db_range = gain_db_range
        self.p_gain = p_gain
        self.rng = random.Random(seed)

    def __call__(self, wav: Tensor) -> Tensor:
        r = self.rng
        if self.p_speed > 0 and r.random() < self.p_speed:
            wav = speed_perturb(wav, r.choice(self.speed_factors), self.sample_rate)
        if self.rir_sampler is not None and r.random() < self.p_rir:
            wav = apply_rir(wav, self.rir_sampler())
        if self.noise_sampler is not None and r.random() < self.p_noise:
            snr = r.uniform(*self.snr_db_range)
            wav = add_noise(wav, self.noise_sampler(), snr, r)
        if self.p_gain > 0 and r.random() < self.p_gain:
            wav = gain_perturb(wav, r.uniform(*self.gain_db_range))
        return wav

    @classmethod
    def from_dirs(
        cls,
        musan_dir: str | None,
        rir_dir: str | None,
        sample_rate: int = 16000,
        seed: int | None = None,
        rir_pack: str | None = None,
        **kwargs,
    ) -> "WaveformAugment":
        rng = random.Random(seed)

        def _load(path: Path) -> Tensor:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(data).mean(dim=1)
            if sr != sample_rate:
                wav = torchaudio.functional.resample(wav, sr, sample_rate)
            return wav

        noise_sampler = None
        if musan_dir:
            noises = sorted(Path(musan_dir).rglob("*.wav"))
            if noises:
                noise_sampler = lambda: _load(rng.choice(noises))  # noqa: E731

        rir_sampler = None
        if rir_pack:
            # FUSE-proof: RIRs live in RAM, no per-sample I/O.
            from .rir_pack import load_rir_pack

            rirs_mem = load_rir_pack(rir_pack, sample_rate=sample_rate)
            if rirs_mem:
                rir_sampler = lambda: rng.choice(rirs_mem)  # noqa: E731
        elif rir_dir:
            rirs = sorted(Path(rir_dir).rglob("*.wav"))
            if rirs:
                rir_sampler = lambda: _load(rng.choice(rirs))  # noqa: E731

        return cls(noise_sampler, rir_sampler, sample_rate=sample_rate, seed=seed, **kwargs)
