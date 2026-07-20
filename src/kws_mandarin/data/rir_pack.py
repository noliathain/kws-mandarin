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
              sample_rate: int = 16000, seed: int = 0) -> int:
    """Read up to ``max_rirs`` impulse responses from ``rir_dir`` into a single .pt file."""
    rng = random.Random(seed)
    paths = sorted(Path(rir_dir).rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no RIR wavs under {rir_dir}")
    if len(paths) > max_rirs:
        paths = rng.sample(paths, max_rirs)
    rirs: list[Tensor] = []
    for p in paths:
        try:
            r = _load(p, sample_rate)
        except Exception:
            continue
        if r.numel() > 0:
            rirs.append(r)
    if not rirs:
        raise RuntimeError(f"no readable RIRs in {rir_dir}")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack RIR wavs into a single in-memory file")
    ap.add_argument("--rir-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rirs", type=int, default=2000)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n = pack_rirs(args.rir_dir, args.out, args.max_rirs, args.sample_rate, args.seed)
    print(f"packed {n} RIRs -> {args.out}")


if __name__ == "__main__":
    main()
