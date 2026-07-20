# kws-mandarin

Robust, low-parameter **open-vocabulary Mandarin keyword spotting (KWS)**.

The design thesis (and the one non-negotiable idea in this repo): **robustness is won in
the decoder and the data, not the backbone.** The acoustic backbone is nearly solved at
our parameter budget — BC-ResNet, small Conformer, and Mamba all land within a point or
two — so we spend our effort on the CTC decoder, noise-aware training, and the evaluation
protocol, not on chasing a fancier encoder.

## Architecture

```
audio ─► log-mel ─► BC-ResNet encoder ─► CTC head over tonal-pinyin units
                                              │
                    (noise-aware training: NTC-style wildcard arcs)
                                              │
                                              ▼
                              open-vocab keyword search
                         (plain CTC now; MFA-KWS TDT head later)
```

- **Backbone:** BC-ResNet (broadcasted residual blocks + SubSpectralNorm), scale 3–6.
  Chosen for its proven int8 quantization / export path, not for raw accuracy.
- **Units:** tonal-pinyin, **compositional so any keyword is expressible** — default
  `initial + toned-final` (~300 units). Configurable to `separate-tone` (~85) or full
  `syllable` (~1700) for ablation. See [docs/architecture.md](docs/architecture.md).
- **Robustness:** NTC-style wildcard-arc CTC (self-loop = noise insertion, bypass =
  masking), multi-SNR MUSAN/RIR augmentation, LLM-generated tone-confusable hard negatives.
- **Open vocabulary:** keywords are token sequences resolved at runtime. Plain CTC keyword
  search ships first; the MFA-KWS Token-and-Duration Transducer head slots in behind the
  same decoder interface without a rewrite.

## What this repo deliberately defers

The deployment target is **not yet chosen**, so nothing here assumes a device. Two
techniques from the original brief are **out of the critical path** until a tier is picked:

- **AdaKWS test-time adaptation** — requires gradient updates at deployment; impossible on
  an inference-only microNPU (e.g. Ethos-U55). It is a phone/edge-Linux-tier refinement.
- **k2 / icefall WFST stack** — heavy dependency. The NTC wildcard arcs are cheap enough to
  implement directly; we adopt k2 only if a decision genuinely needs general WFST machinery.

See [docs/decisions.md](docs/decisions.md) for the full rationale (ADR-style).

## Status

| Component                         | State        |
|-----------------------------------|--------------|
| Tonal-pinyin tokenizer            | ✅ implemented + tested |
| FRR@FAH / DET evaluation harness  | ✅ implemented + tested |
| Log-mel frontend                  | ✅ implemented + tested |
| BC-ResNet encoder + SubSpectralNorm | ✅ implemented + tested |
| CTC loss (plain)                  | ✅ implemented + tested |
| NTC-style wildcard-arc CTC (noise-aware decode) | ✅ implemented + tested |
| Open-vocab keyword decoder        | ✅ implemented + tested |
| MUSAN/RIR + SpecAugment augmentation | ✅ implemented + tested |
| Data prep (AISHELL-1 manifests)   | ✅ implemented + tested |
| Data-quality validation (fail-loud) | ✅ implemented + tested |
| Dataset + collate + CTC training step | ✅ implemented + tested |
| DDP trainer (AMP, ckpt/resume, EMA, FRR@FAH val) | ✅ implemented + tested |
| WebDataset shards (FUSE/S3-robust I/O) | ✅ implemented + tested |
| Tone-confusable hard negatives (LLM-source drop-in) | ✅ implemented + tested |
| RIR-packer (FUSE-proof in-memory reverb) | ✅ implemented + tested |

## Setup

```bash
uv sync --extra dev          # light env: tokenizer + eval + tests, no torch
uv run pytest                # run the test suite
uv sync --extra train        # add torch/torchaudio when building the model
```

## Training

```bash
# 1. corpora (AISHELL-1 + MUSAN + RIRs) -> $KWS_DATA_ROOT
KWS_DATA_ROOT=/teamspace/lightning_storage/kws-mandarin scripts/download_data.sh all

# 2. manifests (96-way) + fail-loud quality gate
uv run python -m kws_mandarin.data.prepare_aishell --aishell-root <root>/corpora/aishell1/data_aishell --out <root>/manifests --workers $(nproc)
uv run python -m kws_mandarin.data.validate --manifests <root>/manifests --deep --workers $(nproc)

# 3. pack audio into WebDataset shards (FUSE/S3-friendly; avoids 141k small-file reads)
uv run python -m kws_mandarin.data.shard --manifest <root>/manifests/aishell1_train.jsonl --out <root>/shards/train --num-shards 256 --workers $(nproc)

# 4. train on 8 GPUs (DDP)  — set data.train_shards to the shard glob in the config.
# Run torchrun THROUGH uv (--extra train) so it uses the .venv python, not system/conda python.
uv run --extra train torchrun --standalone --nproc_per_node=8 -m kws_mandarin.train --config configs/base.yaml
```

Set `data.train_shards: <root>/shards/train/*.tar` in the config to stream from shards; the
`ShardDataset` partitions shards across DDP ranks and dataloader workers (no duplication) with
a sample shuffle buffer. Leave it empty to read the manifest directly (fine for local disk).

Config lives in [configs/base.yaml](configs/base.yaml). Validation reports token error
rate and **FRR@{0.5,1.0} FA/hr** on held-out keywords; best/latest checkpoints and full
resume state are written to `ckpt_dir`.

## Layout

```
src/kws_mandarin/
  tokenizer/    tonal-pinyin → compositional units (implemented)
  eval/         FRR@FAH, DET curves (implemented)
docs/
  architecture.md   models, units, decoder
  decisions.md      ADR-style rationale for every non-obvious call
  evaluation.md     the FRR@FAH protocol — read before trusting any number
  data.md           datasets + license status (commercial-use flags)
tests/
```
