"""Rigorous open-vocab KWS benchmark: EER / AUC / recall@FAH on a large curated trial list.

Replaces the 500-utt training-time validation with a proper evaluation on the held-out TEST
set (speaker-disjoint), following the LibriPhrase protocol: Easy and Hard negative splits,
threshold-free EER + AUC, and bootstrap confidence intervals over keywords so we can finally
say whether one model's edge over another is real or noise.

    uv run python scripts/benchmark_kws.py \
        --ckpt .../checkpoints/sweep_s20/best.pt --config configs/sweep_s20.yaml \
        --out reports/bench_s20.json

Comparability note: metrics and protocol match the open-vocab KWS literature (EER/AUC on
Easy/Hard), but this is AISHELL Mandarin, not English LibriPhrase — the same axis, not the
same dataset. A number ON LibriPhrase needs an English-trained model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kws_mandarin.config import TrainConfig
from kws_mandarin.data.manifest import read_manifest
from kws_mandarin.decode.keyword_search import CTCKeywordSpotter, NTCKeywordSpotter
from kws_mandarin.eval import auc, eer, summary
from kws_mandarin.eval.benchmark import build_trials, select_keywords
from kws_mandarin.models import KWSModel
from kws_mandarin.tokenizer import PinyinTokenizer, ToneMode
from kws_mandarin.train.validate_kws import _load_wav


def _score_all(model, utts, trials, tok, device, sr, batch_size=32):
    """Score every trial. Each utterance's log-probs are computed once and reused across all
    queries that target it (positives + its hard negatives), so cost is one forward per utt.
    """
    blank = tok.blank_id
    ctc = CTCKeywordSpotter(blank=blank)
    by_utt: dict[str, list[int]] = {}
    for i, t in enumerate(trials):
        by_utt.setdefault(t.utt_id, []).append(i)
    utt_by_id = {u.utt_id: u for u in utts}
    scores = [0.0] * len(trials)

    ids = [uid for uid in by_utt if uid in utt_by_id]
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        wavs = [_load_wav(utt_by_id[uid].wav, sr) for uid in chunk]
        lengths = torch.tensor([w.numel() for w in wavs])
        n_max = int(lengths.max())
        batch = torch.zeros(len(wavs), n_max, device=device)
        for i, w in enumerate(wavs):
            batch[i, : w.numel()] = w.to(device)
        log_probs = model(batch, lengths).log_softmax(-1)
        out_lens = model.output_lengths(lengths).to(device)
        for row, uid in enumerate(chunk):
            lp = log_probs[row : row + 1]
            olen = out_lens[row : row + 1]
            for ti in by_utt[uid]:
                q = trials[ti].query_ids
                scores[ti] = float(ctc.score_batch(lp, olen, list(q))[0])
    return scores


def _metrics(scores, labels, durations):
    m = {"eer": round(eer(scores, labels), 4), "auc": round(auc(scores, labels), 4),
         "n_pos": int(sum(labels)), "n_neg": int(len(labels) - sum(labels))}
    for fah, op in summary(scores, labels, durations, (0.5, 1.0)).items():
        m[f"frr@{fah}"] = round(op.frr, 4)
    return m


def _bootstrap_eer(scores, labels, keys, seed=0, n=500):
    """Bootstrap EER by resampling KEYWORDS (not trials) — trials within a keyword are
    correlated, so resampling keywords gives an honest interval. Returns (lo, hi) at 95%.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    by_key: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    uniq = list(by_key)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        idx = [i for p in pick for i in by_key[uniq[p]]]
        s = [scores[i] for i in idx]
        l = [labels[i] for i in idx]
        if any(l) and not all(l):
            vals.append(eer(s, l))
    if not vals:
        return None, None
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/sweep_s20.yaml")
    ap.add_argument("--manifest", default="/teamspace/lightning_storage/kws-mandarin/manifests/aishell1_test.jsonl")
    ap.add_argument("--n-keywords", type=int, default=60)
    ap.add_argument("--max-utts", type=int, default=0, help="0 = use the whole manifest")
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = PinyinTokenizer(ToneMode(cfg.model.tone_mode))

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get(args.weights) or ck["model"]
    model = KWSModel(vocab_size=tok.vocab_size, n_mels=cfg.model.n_mels, scale=cfg.model.scale,
                     causal=cfg.model.causal, ssn_bands=cfg.model.ssn_bands, dropout=0.0,
                     blank_id=tok.blank_id).to(device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    model.eval()

    utts = read_manifest(args.manifest)
    if args.max_utts:
        utts = utts[: args.max_utts]
    keywords = select_keywords(utts, tok, n=args.n_keywords)
    trials = build_trials(utts, keywords, tok, seed=cfg.seed)
    print(f"{len(keywords)} keywords, {len(trials)} trials over {len(utts)} utts", flush=True)

    with torch.no_grad():
        scores = _score_all(model, utts, trials, tok, device, cfg.data.sample_rate)

    report = {"ckpt": args.ckpt, "scale": cfg.model.scale, "n_keywords": len(keywords),
              "n_utts": len(utts), "splits": {}}
    # pos vs each negative split, and pos vs all negatives
    for split in ("easy", "hard", "all"):
        sel = [i for i, t in enumerate(trials)
               if t.label == 1 or (split == "all") or t.split == split]
        s = [scores[i] for i in sel]
        l = [trials[i].label for i in sel]
        d = [trials[i].duration for i in sel]
        k = [trials[i].query for i in sel]
        m = _metrics(s, l, d)
        m["eer_ci95"] = _bootstrap_eer(s, l, k, seed=cfg.seed)
        report["splits"][split] = m
        ci = m["eer_ci95"]
        print(f"  {split:5s}  EER={m['eer']:.4f} [{ci[0]}, {ci[1]}]  AUC={m['auc']:.4f}  "
              f"frr@1.0={m.get('frr@1.0')}  (pos {m['n_pos']}, neg {m['n_neg']})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
