# Open-vocabulary Mandarin KWS — model capacity sweep

**Question.** At a fixed training recipe, how far does keyword-spotting accuracy scale
with model size — and where do the returns stop?

**Answer.** Capacity was the binding constraint. Token error rate drops 43% over the first
~30× of parameters, then saturates: scale 30 and scale 40 tie on TER despite scale 40 having
1.7× the parameters. The knee of the curve sits near **2M parameters**.

An interactive version of these plots is in [`capacity_sweep.html`](capacity_sweep.html).

## Setup

All six models are the same architecture (BC-ResNet encoder → CTC over compositional
tonal-syllable units), trained with an **identical recipe** — only `model.scale` differs:

| Setting | Value |
|---|---|
| Data | AISHELL-1, 120,098 train utts, PCM shards |
| Steps | 15,000 |
| Global batch | 512 (micro-batch 64 × accum 4 × 2 GPUs) |
| LR | 3e-3, 1000-step warmup, cosine decay to 1% |
| Precision | bf16 with EMA (warmup-ramped decay) |
| Augmentation | RIR + MUSAN noise + speed perturbation (on GPU), SpecAugment |
| Hardware | 2× H100 per run, several runs in parallel across boxes |

Global batch is held at 512 for **every** point via micro-batch 64 + gradient accumulation, so
BatchNorm sees the same statistics at every scale. This matters: an earlier partial sweep used
micro-batches of 256/256/64 and was **not** comparable — capacity and BN were changing together.
Those runs were redone at micro-batch 64 before drawing any conclusion.

## Results

| scale | params | TER | FRR @0.5/hr | FRR @1/hr | it/s |
|------:|-------:|----:|------------:|----------:|-----:|
| 3.0  | 62,260    | 0.4611 | 0.550 | 0.4667 | 4.9 |
| 6.0  | 189,880   | 0.3705 | 0.483 | 0.4000 | 4.2 |
| 12.0 | 642,112   | 0.3129 | 0.400 | 0.2833 | 3.1 |
| 20.0 | 1,653,664 | 0.2860 | 0.500 | 0.2167 | 2.3 |
| 30.0 | 3,574,744 | **0.2650** | 0.383 | 0.2500 | 1.7 |
| 40.0 | 6,225,424 | **0.2649** | 0.400 | **0.1833** | 1.3 |

(TER = token error rate on the full dev set; FRR@N/hr = false-reject rate at N false alarms
per hour, on 500 dev utts × 10 keywords.)

## Reading the curve

- **TER is the reliable signal.** Measured over the full dev set, it falls monotonically with
  size and flattens hard: 0.286 → 0.265 → 0.265 across the last three scales. Scale 40 buys
  essentially nothing over scale 30 on TER.
- **FRR is noisy.** 500 utterances × 10 keywords is a small denominator, so single-point
  wobbles (scale 30's 0.25, worse than scale 20's 0.217) are within noise. Trust the trend,
  not the point.
- **Every curve is still descending at 15,000 steps.** These are lower bounds, not converged
  ceilings — a longer schedule shifts the whole curve down. The *shape* (the knee) should hold.

## Recommendation

For a low-parameter deployment target, **scale 12–20 (642k–1.65M params)** is the sweet spot:
most of the achievable accuracy at a fraction of the largest model, and cheap to train
(scale 20 at 2.3 it/s). Scale 40 only justifies its cost if FRR@1/hr is the optimization
target and the noise there proves real on a larger dev set.

## Reproducing

```bash
# one config per box; each writes its own checkpoints + timestamped log
bash scripts/train.sh 2 configs/sweep_s3.yaml  --fresh
bash scripts/train.sh 2 configs/sweep_s6.yaml  --fresh
bash scripts/train.sh 2 configs/sweep_s12.yaml --fresh
bash scripts/train.sh 2 configs/sweep_s20.yaml --fresh
bash scripts/train.sh 2 configs/sweep_s30.yaml --fresh
bash scripts/train.sh 2 configs/sweep_s40.yaml --fresh   # micro 64; sweep_s40_m32.yaml if it OOMs

# aggregate every run from the shared log dir (recovers scale from each log header)
uv run python scripts/collect_sweep.py --out sweep.json
```

## Caveats worth carrying forward

1. Dev set is small for FRR (500 × 10). A larger keyword panel would tighten the FRR column.
2. 15,000 steps is short; the accuracy floor per scale is lower than reported here.
3. The sweep varies width only (`model.scale`). Depth, tone-unit granularity (final vs
   separate vs syllable), and the NTC noise-aware decoder are unexplored axes.
