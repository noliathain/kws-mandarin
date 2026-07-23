"""Zero-shot evaluation on MobvoiHotwords (OpenSLR 87) — the standard Mandarin KWS benchmark.

Our open-vocab model never trains on Mobvoi; it scores each utterance against the hotword's
tonal-unit sequence and the threshold is swept. Reports recall@FA/hr (the Mobvoi convention),
plus EER and AUC, per hotword.

    # one-time download (~2 GB audio + metadata) on the GPU box:
    cd <scratch> && mkdir -p mobvoi && cd mobvoi
    curl -LO https://www.openslr.org/resources/87/mobvoi_hotword_dataset.tgz
    curl -LO https://www.openslr.org/resources/87/mobvoi_hotword_dataset_resources.tgz
    tar xzf mobvoi_hotword_dataset.tgz            # -> *.wav
    tar xzf mobvoi_hotword_dataset_resources.tgz  # -> mobvoi_hotword_dataset_resources/*.json

    uv run python scripts/eval_mobvoi.py \
        --ckpt .../checkpoints/sweep_s20/best.pt --config configs/sweep_s20.yaml \
        --mobvoi-root <scratch>/mobvoi --out /teamspace/lightning_storage/kws-mandarin/bench/mobvoi_s20.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from kws_mandarin.config import TrainConfig
from kws_mandarin.decode.keyword_search import CTCKeywordSpotter
from kws_mandarin.eval import auc, eer, summary
from kws_mandarin.eval.mobvoi import HOTWORDS, build_mobvoi_trials
from kws_mandarin.models import KWSModel
from kws_mandarin.tokenizer import PinyinTokenizer, ToneMode


def _find_audio(root: Path) -> dict[str, Path]:
    """Map utt_id -> wav path. The audio tarball lays wavs out flat or one level down; we index
    once so a missing file fails loudly with a clear message rather than mid-scoring.
    """
    idx = {}
    for p in root.rglob("*.wav"):
        idx[p.stem] = p
    if not idx:
        raise FileNotFoundError(f"no .wav files under {root} — did you extract the audio tarball?")
    return idx


def _load(path: Path, sr: int) -> torch.Tensor:
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    t = torch.from_numpy(wav).mean(dim=1)
    if file_sr != sr:
        import torchaudio
        t = torchaudio.functional.resample(t, file_sr, sr)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/sweep_s20.yaml")
    ap.add_argument("--mobvoi-root", required=True, help="dir holding the wavs + resources json")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--max-neg", type=int, default=0, help="cap negatives for speed (0 = all)")
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr = cfg.data.sample_rate
    tok = PinyinTokenizer(ToneMode(cfg.model.tone_mode))
    ctc = CTCKeywordSpotter(blank=tok.blank_id)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get(args.weights) or ck["model"]
    model = KWSModel(vocab_size=tok.vocab_size, n_mels=cfg.model.n_mels, scale=cfg.model.scale,
                     causal=cfg.model.causal, ssn_bands=cfg.model.ssn_bands, dropout=0.0,
                     blank_id=tok.blank_id).to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    model.eval()

    root = Path(args.mobvoi_root)
    res = root / "mobvoi_hotword_dataset_resources"
    p_entries = json.loads((res / f"p_{args.split}.json").read_text())
    n_entries = json.loads((res / f"n_{args.split}.json").read_text())
    if args.max_neg:
        n_entries = n_entries[: args.max_neg]
    audio = _find_audio(root)

    print(f"hotword map: {HOTWORDS}  (verify against the corpus README)", flush=True)
    report = {"ckpt": args.ckpt, "scale": cfg.model.scale, "split": args.split,
              "weights": args.weights, "hotwords": {}}

    for kid, text in HOTWORDS.items():
        kw_ids = tok.encode(text)
        trials = build_mobvoi_trials(p_entries, n_entries, kid)
        trials = [(u, l) for u, l in trials if u in audio]     # skip any missing wavs

        # score every trial (batched, one forward per utterance)
        scores, labels, durs = [], [], []
        for start in range(0, len(trials), args.batch_size):
            chunk = trials[start : start + args.batch_size]
            wavs = [_load(audio[u], sr) for u, _ in chunk]
            lengths = torch.tensor([w.numel() for w in wavs])
            n_max = int(lengths.max())
            batch = torch.zeros(len(wavs), n_max, device=device)
            for i, w in enumerate(wavs):
                batch[i, : w.numel()] = w.to(device)
            with torch.no_grad():
                lp = model(batch, lengths).log_softmax(-1)
            out_lens = model.output_lengths(lengths).to(device)
            s = ctc.score_batch(lp, out_lens, kw_ids).cpu()
            for i, (_, lab) in enumerate(chunk):
                scores.append(float(s[i])); labels.append(lab)
                durs.append(wavs[i].numel() / sr)
            if start % (args.batch_size * 50) == 0:
                print(f"  [{text}] {start}/{len(trials)}", flush=True)

        m = {"eer": round(eer(scores, labels), 4), "auc": round(auc(scores, labels), 4),
             "n_pos": int(sum(labels)), "n_neg": int(len(labels) - sum(labels))}
        for fah, op in summary(scores, labels, durs, (0.5, 1.0)).items():
            m[f"recall@{fah}fa/h"] = round(1.0 - op.frr, 4)
        report["hotwords"][text] = m
        print(f"  {text}: EER={m['eer']:.4f} AUC={m['auc']:.4f} "
              f"recall@0.5={m['recall@0.5fa/h']} recall@1.0={m['recall@1.0fa/h']} "
              f"(pos {m['n_pos']}, neg {m['n_neg']})", flush=True)

    macro = {k: round(sum(h[k] for h in report["hotwords"].values()) / len(HOTWORDS), 4)
             for k in ("eer", "auc", "recall@0.5fa/h", "recall@1.0fa/h")}
    report["macro"] = macro
    print(f"  MACRO: EER={macro['eer']} AUC={macro['auc']} "
          f"recall@0.5={macro['recall@0.5fa/h']} recall@1.0={macro['recall@1.0fa/h']}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
