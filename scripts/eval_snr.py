"""Noise-robustness curve: FRR@FAH and TER vs SNR for a trained KWS checkpoint.

We TRAIN with additive noise but never measured robustness — this attaches a number to it.
For each SNR level the whole dev set is corrupted at that SNR and scored through the normal
validation path, so the only variable is noise level.

    uv run python scripts/eval_snr.py \
        --ckpt /teamspace/lightning_storage/kws-mandarin/checkpoints/sweep_s20/best.pt \
        --out reports/snr_sweep_s20.json

Caveat, stated loudly: the noise here is MUSAN, the SAME corpus the model trained on. This
measures robustness to the training noise DISTRIBUTION, not to unseen noise — a strong result
here is necessary but not sufficient. A truly held-out source (WHAM!/DEMAND, as MFA-KWS uses)
is the honest follow-up; pass --noise-pack to point at one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kws_mandarin.config import TrainConfig
from kws_mandarin.data.augment import add_noise_batch
from kws_mandarin.data.manifest import read_manifest
from kws_mandarin.data.rir_pack import load_noise_pack
from kws_mandarin.models import KWSModel
from kws_mandarin.tokenizer import PinyinTokenizer, ToneMode
from kws_mandarin.train.validate_kws import run_validation


def _make_noise_bank(pack_path: str, sample_rate: int, length: int, device) -> torch.Tensor:
    """Tile every noise clip to a common length so any batch has a valid random crop."""
    clips = load_noise_pack(pack_path, sample_rate=sample_rate)
    bank = torch.zeros(len(clips), length)
    for i, n in enumerate(clips):
        if n.numel() == 0:
            continue
        reps = (length + n.numel() - 1) // n.numel()
        bank[i] = n.repeat(reps)[:length]
    return bank.to(device)


def _corrupt_fn(bank: torch.Tensor, snr_db: float, seed: int):
    """Return a corrupt_fn that mixes a random noise crop at exactly ``snr_db`` per utterance.

    Deterministic given seed+snr, so the clean/noisy runs differ ONLY in the noise, never in
    which crop was drawn — the SNR comparison is otherwise identical.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + int(snr_db * 100))

    def fn(wavs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, t = wavs.shape
        bank_n, bank_len = bank.shape
        idx = torch.randint(0, bank_n, (b,), generator=gen).to(wavs.device)
        max_off = max(1, bank_len - t + 1)
        off = torch.randint(0, max_off, (b,), generator=gen).to(wavs.device)
        pos = (off.unsqueeze(1) + torch.arange(t, device=wavs.device)).clamp(max=bank_len - 1)
        noise = bank[idx].gather(1, pos)
        snr = torch.full((b,), float(snr_db), device=wavs.device)
        return add_noise_batch(wavs, noise, snr, lengths.to(wavs.device))

    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/sweep_s20.yaml",
                    help="config for tokenizer/model geometry and dev manifest")
    ap.add_argument("--noise-pack", default="/teamspace/lightning_storage/kws-mandarin/musan.pt")
    ap.add_argument("--snrs", default="clean,20,15,10,5,0")
    ap.add_argument("--max-utts", type=int, default=2000)
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = PinyinTokenizer(ToneMode(cfg.model.tone_mode))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get(args.weights) or ck["model"]
    model = KWSModel(
        vocab_size=tok.vocab_size, n_mels=cfg.model.n_mels, scale=cfg.model.scale,
        causal=cfg.model.causal, ssn_bands=cfg.model.ssn_bands, dropout=0.0,
        blank_id=tok.blank_id,
    ).to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    model.eval()

    utts = read_manifest(cfg.data.dev_manifest)
    sr = cfg.data.sample_rate
    length = int(cfg.data.max_duration_s * sr)
    bank = _make_noise_bank(args.noise_pack, sr, length, device)

    rows = []
    for tok_s in args.snrs.split(","):
        tok_s = tok_s.strip()
        clean = tok_s.lower() == "clean"
        cf = None if clean else _corrupt_fn(bank, float(tok_s), seed=cfg.seed)
        m = run_validation(
            model, utts, tok, cfg.val_keywords, device,
            sample_rate=sr, max_utts=args.max_utts,
            use_ntc=cfg.val_use_ntc, ntc_lambda=cfg.ntc_lambda, corrupt_fn=cf,
        )
        row = {"snr": "clean" if clean else float(tok_s), **{k: round(v, 4) for k, v in m.items()}}
        rows.append(row)
        print(f"  SNR {tok_s:>5}  ter={row.get('ter'):.4f}  "
              f"frr@0.5={row.get('frr@0.5')}  frr@1.0={row.get('frr@1.0')}", flush=True)

    out = {"ckpt": args.ckpt, "scale": cfg.model.scale, "noise_pack": args.noise_pack,
           "weights": args.weights, "max_utts": args.max_utts, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
