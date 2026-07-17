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
  `initial + toned-final` (~280 units). Configurable to `separate-tone` (~85) or full
  `syllable` (~1380) for ablation. See [docs/architecture.md](docs/architecture.md).
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
| Log-mel frontend                  | ⬜ next |
| BC-ResNet encoder + SubSpectralNorm | ⬜ next |
| CTC loss + wildcard-arc training  | ⬜ planned |
| Open-vocab keyword decoder        | ⬜ planned |
| MUSAN/RIR augmentation            | ⬜ next (corpora downloader ready) |
| Data prep (AISHELL-1 manifests)   | ✅ implemented + tested |

## Setup

```bash
uv sync --extra dev          # light env: tokenizer + eval + tests, no torch
uv run pytest                # run the test suite
uv sync --extra train        # add torch/torchaudio when building the model
```

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
