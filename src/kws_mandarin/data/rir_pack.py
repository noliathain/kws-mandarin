"""Pack a RIR subset into one file loaded into RAM — FUSE/S3-proof reverb augmentation.

Reading individual RIR wavs off an S3-backed FUSE mount per training sample would bottleneck
the run (and extracting all ~60k of them is impractical there). Instead we read a bounded set
of impulse responses ONCE, resample to the model rate, and torch.save them as a single file.
Training loads that one file into memory at startup; augmentation samples RIRs from RAM with
zero per-sample I/O.

    uv run python -m kws_mandarin.data.rir_pack \\
        --rir-dir <root>/corpora/rirs/RIRS_NOISES/real_rirs_isotropic_noises \\
        --out     <root>/rirs.pt --max-rirs 2000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch import Tensor


def _load(path: Path, sample_rate: int) -> Tensor:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def pack_rirs(rir_dir: str, out_path: str, max_rirs: int = 2000,
              sample_rate: int = 16000, seed: int = 0, max_rir_seconds: float = 2.0) -> int:
    """Read up to ``max_rirs`` impulse responses from ``rir_dir`` into a single .pt file.

    Files longer than ``max_rir_seconds`` are skipped: an "impulse response" more than a
    couple of seconds long is not a RIR but an isotropic-noise recording (OpenSLR 28 ships
    both in the same folder), and convolving with one is both wrong and ruinously expensive.
    """
    rng = random.Random(seed)
    paths = sorted(Path(rir_dir).rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no RIR wavs under {rir_dir}")
    if len(paths) > max_rirs:
        paths = rng.sample(paths, max_rirs)
    max_len = int(max_rir_seconds * sample_rate)
    rirs: list[Tensor] = []
    skipped = 0
    for p in paths:
        try:
            r = _load(p, sample_rate)
        except Exception:
            continue
        if r.numel() == 0 or r.numel() > max_len:
            skipped += 1  # empty, or too long to be a real RIR (isotropic noise)
            continue
        rirs.append(r)
    if not rirs:
        raise RuntimeError(f"no readable RIRs in {rir_dir}")
    if skipped:
        print(f"skipped {skipped} non-RIR files (> {max_rir_seconds}s or empty)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rirs": rirs, "sample_rate": sample_rate}, out_path)
    return len(rirs)


def load_rir_pack(path: str, sample_rate: int | None = None) -> list[Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if sample_rate is not None and obj.get("sample_rate") != sample_rate:
        raise ValueError(
            f"RIR pack sample_rate {obj.get('sample_rate')} != expected {sample_rate}"
        )
    return obj["rirs"]


def pack_noise(noise_dir: str, out_path: str, max_items: int = 1000,
               sample_rate: int = 16000, max_seconds: float = 15.0, seed: int = 0) -> int:
    """Pack a MUSAN-style noise subset into one .pt loaded into RAM (FUSE-proof augmentation).

    Same rationale as ``pack_rirs``: reading thousands of small noise files off an S3-FUSE
    mount per training sample stalls the input pipeline. Long clips are cropped to
    ``max_seconds`` (``add_noise`` tiles/crops to the utterance length anyway) to bound RAM.
    """
    rng = random.Random(seed)
    paths = sorted(Path(noise_dir).rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no noise wavs under {noise_dir}")
    if len(paths) > max_items:
        paths = rng.sample(paths, max_items)
    max_len = int(max_seconds * sample_rate)
    noises: list[Tensor] = []
    for p in paths:
        try:
            r = _load(p, sample_rate)
        except Exception:
            continue
        if r.numel() == 0:
            continue
        noises.append(r[:max_len] if r.numel() > max_len else r)
    if not noises:
        raise RuntimeError(f"no readable noise in {noise_dir}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"noises": noises, "sample_rate": sample_rate}, out_path)
    return len(noises)


def load_noise_pack(path: str, sample_rate: int | None = None) -> list[Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if sample_rate is not None and obj.get("sample_rate") != sample_rate:
        raise ValueError(f"noise pack sample_rate {obj.get('sample_rate')} != expected {sample_rate}")
    return obj["noises"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack RIR/noise wavs into a single in-memory file")
    ap.add_argument("--noise", action="store_true", help="pack noise (crop long) instead of RIRs")
    ap.add_argument("--rir-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rirs", type=int, default=2000)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rir-seconds", type=float, default=2.0,
                    help="RIR mode: skip files longer than this (isotropic noise, not RIRs)")
    ap.add_argument("--max-noise-seconds", type=float, default=15.0,
                    help="noise mode: crop clips longer than this")
    args = ap.parse_args()
    if args.noise:
        n = pack_noise(args.rir_dir, args.out, args.max_rirs, args.sample_rate,
                       args.max_noise_seconds, args.seed)
        print(f"packed {n} noise clips -> {args.out}")
    else:
        n = pack_rirs(args.rir_dir, args.out, args.max_rirs, args.sample_rate, args.seed,
                      args.max_rir_seconds)
        print(f"packed {n} RIRs -> {args.out}")


if __name__ == "__main__":
    main()
