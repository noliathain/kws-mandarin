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
from fractions import Fraction
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
    aligned to the RIR's direct-path peak and trimmed back to the input length. Uses FFT
    convolution so a long RIR does not blow up memory (a plain conv1d with a multi-second
    kernel tries to allocate hundreds of GB).
    """
    n = wav.numel()
    energy = rir.pow(2).sum().sqrt()
    if energy <= 0:
        return wav
    rir = rir / energy
    peak = int(torch.argmax(rir.abs()))
    conv = torchaudio.functional.fftconvolve(wav, rir)  # full convolution, (n + L - 1,)
    return conv[peak : peak + n]


def apply_rir_batch(wavs: Tensor, rirs: Tensor) -> Tensor:
    """Batched RIR convolution on-device (for GPU augmentation).

    ``wavs`` (B, T), ``rirs`` (B, L) zero-padded impulse responses -> (B, T). Each waveform is
    convolved with its own RIR via a grouped conv1d (B independent convolutions in parallel),
    unit-energy normalized and aligned to the RIR's direct-path peak, length preserved. This
    runs in the training step so DataLoader workers (which crash on this container's shm IPC)
    aren't needed for the costly reverb augmentation.
    """
    b, t = wavs.shape
    length = rirs.shape[1]
    energy = rirs.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    rirs = rirs / energy
    out = torch.nn.functional.conv1d(
        wavs.unsqueeze(0),                # (1, B, T)
        rirs.flip(1).unsqueeze(1),        # (B, 1, L) flipped -> convolution
        padding=length - 1, groups=b,
    ).squeeze(0)                          # (B, T + L - 1)
    peaks = rirs.abs().argmax(dim=1)                                    # (B,)
    idx = peaks.unsqueeze(1) + torch.arange(t, device=wavs.device)      # (B, T)
    return out.gather(1, idx)


def speed_perturb(wav: Tensor, factor: float, sample_rate: int = 16000) -> Tensor:
    """Speed up (factor>1) / slow down (<1) by resampling. Changes duration and pitch.

    Resamples with a SMALL integer ratio. Passing (sample_rate, sample_rate/factor) directly
    makes torchaudio build a huge sinc filter for near-coprime pairs (e.g. 16000 -> 14545),
    which is pathologically slow. A speed-up by ``factor`` gives new_len = N/factor, i.e.
    resample(orig=numerator, new=denominator) for factor = numerator/denominator.
    """
    if factor == 1.0:
        return wav
    frac = Fraction(factor).limit_denominator(20)  # 1.1 -> 11/10, 0.9 -> 9/10
    return torchaudio.functional.resample(wav, frac.numerator, frac.denominator)


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

class _MemSampler:
    """Picklable random-choice over an in-memory list (safe for forkserver/spawn workers)."""

    def __init__(self, items: list[Tensor], seed: int | None = None):
        self._items = items
        self._rng = random.Random(seed)

    def __call__(self) -> Tensor:
        return self._items[self._rng.randrange(len(self._items))]


class _PackSampler:
    """Picklable, lazily-loaded sampler over an audio pack (.pt).

    Stores only the pack *path*; loads the tensors inside each worker on first use. This keeps
    large tensor lists out of the fork/pickle boundary — pickling hundreds of shared-memory
    tensors trips forkserver's "too many fds" limit.
    """

    def __init__(self, pack_path: str, key: str, sample_rate: int, seed: int | None = None):
        self._path = pack_path
        self._key = key
        self._sr = sample_rate
        self._seed = seed
        self._items: list[Tensor] | None = None
        self._rng: random.Random | None = None

    def __call__(self) -> Tensor:
        if self._items is None:
            obj = torch.load(self._path, map_location="cpu", weights_only=False)
            if obj.get("sample_rate") != self._sr:
                raise ValueError(f"pack sample_rate {obj.get('sample_rate')} != {self._sr}")
            self._items = obj[self._key]
            self._rng = random.Random(self._seed)
        return self._items[self._rng.randrange(len(self._items))]


class _FileSampler:
    """Picklable random wav loader (used when augmentation reads from a directory)."""

    def __init__(self, paths, sample_rate: int, seed: int | None = None):
        self._paths = [str(p) for p in paths]
        self._sr = sample_rate
        self._rng = random.Random(seed)

    def __call__(self) -> Tensor:
        p = self._paths[self._rng.randrange(len(self._paths))]
        data, sr = sf.read(p, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data).mean(dim=1)
        if sr != self._sr:
            wav = torchaudio.functional.resample(wav, sr, self._sr)
        return wav


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
        noise_pack: str | None = None,
        **kwargs,
    ) -> "WaveformAugment":
        rir_seed = None if seed is None else seed + 1

        noise_sampler = None
        if noise_pack:  # in-memory noise, loaded lazily per worker
            if not Path(noise_pack).exists():
                raise FileNotFoundError(f"noise_pack not found: {noise_pack}")
            noise_sampler = _PackSampler(noise_pack, "noises", sample_rate, seed)
        elif musan_dir:
            paths = sorted(Path(musan_dir).rglob("*.wav"))
            if paths:
                noise_sampler = _FileSampler(paths, sample_rate, seed)

        rir_sampler = None
        if rir_pack:  # in-memory RIRs, loaded lazily per worker
            if not Path(rir_pack).exists():
                raise FileNotFoundError(f"rir_pack not found: {rir_pack}")
            rir_sampler = _PackSampler(rir_pack, "rirs", sample_rate, rir_seed)
        elif rir_dir:
            paths = sorted(Path(rir_dir).rglob("*.wav"))
            if paths:
                rir_sampler = _FileSampler(paths, sample_rate, rir_seed)

        return cls(noise_sampler, rir_sampler, sample_rate=sample_rate, seed=seed, **kwargs)
