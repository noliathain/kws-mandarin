# Design decisions (ADR-style)

Each entry: the call, the reasoning, and what would change it. These are deliberately
opinionated — the point of a decision record is to make the reasoning contestable later.

## D1 — Backbone: BC-ResNet, not Conformer/Mamba
**Decision.** BC-ResNet (broadcasted residual blocks + SubSpectralNorm), scale 3–6.
**Why.** At our parameter budget the backbone is nearly solved — BC-ResNet, small
Conformer, and Mamba land within ~1–2 points. BC-ResNet wins on the *export* axis: proven
int8 PT2E path, which matters if we ever land on a microNPU. We are not spending our
robustness budget on the encoder.
**Watch.** SubSpectralNorm's per-subband affine has quantization gotchas — fold it into the
preceding conv at export or it breaks PT2E.
**Revisit if.** A device tier is chosen whose operator set favors a different backbone.

## D2 — Robustness lives in the decoder + data, not the backbone
**Decision.** Spend effort on NTC-style wildcard-arc CTC, multi-SNR augmentation, and
confusable hard negatives — not on a bigger encoder.
**Why.** This is the consistent finding across the 2024–2026 KWS literature and the single
most important idea in the repo.

## D3 — Units: compositional tonal-pinyin, default `initial + toned-final`
**Decision.** Default `FINAL` mode (~280 units). `SEPARATE` (~85) and `SYLLABLE` (~1380)
are configurable for ablation. See [architecture.md](architecture.md).
**Why.** Open vocabulary requires a *compositional* inventory so any keyword is expressible.
`FINAL` balances data efficiency against keeping tone attached to its host final (tone is
suprasegmental — detaching it into its own token, as `SEPARATE` does, is acoustically
awkward). `SYLLABLE` matches the CRNN-CTC precedent but is a large softmax.
**Revisit if.** Ablation shows `SEPARATE` matches `FINAL` at lower cost, or `SYLLABLE`
meaningfully wins on tone-confusable pairs.

## D4 — Open vocabulary, decoder behind an interface
**Decision.** Keywords are runtime token sequences. Ship plain CTC keyword search first;
keep the decoder interface stable so the MFA-KWS Token-and-Duration Transducer (~3.3M
params) can slot in later without touching the encoder or tokenizer.
**Why.** The user needs arbitrary keywords, not a fixed set. TDT improves duration modeling
but is not needed to prove the pipeline.

## D5 — Evaluation is FRR @ FAH, built before the model
**Decision.** The DET curve (false-reject rate vs false-alarms-per-hour) is the only metric
that adjudicates design choices. The harness ([../src/kws_mandarin/eval](../src/kws_mandarin/eval))
exists before the encoder.
**Why.** No accuracy scalar is meaningful for KWS. Building the ruler first keeps every
later decision honest.

## D6 — AdaKWS is deferred (out of critical path)
**Decision.** No test-time adaptation until a deployment tier is chosen.
**Why.** AdaKWS needs gradient updates at deployment. That is impossible on an
inference-only microNPU (Ethos-U55) and impractical on an MCU. It is a phone/edge-Linux
refinement, not a core-model feature.
**Revisit if.** Target is phone/edge-Linux.

## D7 — No k2/icefall commitment yet
**Decision.** Implement NTC wildcard arcs directly rather than adopting the k2 WFST stack.
**Why.** k2 is a heavy, hard-to-build dependency. The two arcs (self-loop = noise
insertion, bypass = masking) are cheap to implement standalone.
**Revisit if.** We need general WFST machinery (e.g. rich lexicon/LM composition).

## D8 — Vocabulary is generated and committed, not learned from a corpus
**Decision.** The unit inventory is produced by sweeping the CJK range through pypinyin
([../scripts/build_vocab.py](../scripts/build_vocab.py)) and committed as text files.
**Why.** Open-vocab needs a *fixed, complete* inventory independent of whatever the training
corpus happened to contain. Committing it makes it versioned and reproducible.

## D9 — WenetSpeech is research-only for shipping
**Decision.** WenetSpeech may be used for robustness experiments but never in a shipped
checkpoint. See [data.md](data.md).
**Why.** Its audio is scraped YouTube/Podcast; commercial redistribution rights are not
clean regardless of this repo's license.
