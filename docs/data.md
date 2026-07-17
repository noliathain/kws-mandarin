# Datasets & license status

**License flag first, because this may become product work.** Only Apache-2.0 corpora are
safe in a *shipped* checkpoint. Everything else is research/experiment-only until its terms
are explicitly cleared. Verify each license yourself before it enters a shipping model — the
notes below are guidance, not legal advice.

| Corpus | Size | Role | License / shipping status |
|--------|------|------|---------------------------|
| **AISHELL-1** | ~178 h, 400 spk | **Base acoustic training** | **Apache-2.0 — safe to ship** |
| AISHELL-2 | ~1000 h, 1991 spk | Command-style coverage (reserve) | Academic request — clear before shipping |
| WenetSpeech | 10k+ h | Robustness experiments only | Scraped YouTube/Podcast — **never ship** |
| Aidatatang_200zh | 200 h, 600 spk | Speaker diversity supplement | Check terms |
| KeSpeech | multi | Accent/dialect robustness | Check terms |
| AISHELL-3 | ~85 h, 218 spk | TTS / enrollment synthesis | Check terms |
| AISHELL-4 / AliMeeting | — | Far-field / mic-array (if array target) | Check terms |
| AISHELL-5 | — | In-car hard acoustics (2025) | Check terms |
| Common Voice (zh) | — | Device/accent robustness | CC0 clips / CC-BY — verify |

## Augmentation sources

| Source | Use | Link |
|--------|-----|------|
| MUSAN | additive noise / music / babble | https://www.openslr.org/17/ |
| RIRs | room reverberation | https://www.openslr.org/28/ |
| WHAM! | real-world noise | http://wham.whisper.ai/ |

## Practical recipe

1. Train the encoder on **AISHELL-1** (Apache-2.0) as the clean-license base.
2. Optionally mix in **WenetSpeech** for acoustic diversity **in experiments only** — do not
   let it into a shipped checkpoint.
3. Augment every batch with **MUSAN + RIR** across SNRs down to 0 dB.
4. Generate **tone-confusable hard negatives** synthetically (LLM-Synth4KWS style).
5. Hold **AISHELL-2** in reserve for more command-style coverage if its license clears.

## Reference links (methods)

- BC-ResNet — https://arxiv.org/abs/2106.04140 · https://github.com/Qualcomm-AI-research/bcresnet
- NTC-KWS (wildcard-arc CTC) — https://arxiv.org/abs/2412.12614
- MFA-KWS / TDT-KWS — https://arxiv.org/abs/2505.19577 · https://github.com/X-LANCE/KWStreamingSearch
- AdaKWS (deferred) — https://www.isca-archive.org/interspeech_2025/xiao25b_interspeech.html
- LLM-Synth4KWS (hard negatives) — https://arxiv.org/abs/2505.22995
- Mandarin CRNN-CTC (tonal-syllable units) — https://ieeexplore.ieee.org/document/9054618/

Corpora are never committed to the repo; `data/` is git-ignored.
