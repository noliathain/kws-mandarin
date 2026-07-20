"""Tone-confusable hard negatives (LLM-Synth4KWS's mechanism, done algorithmically).

Tone-confusion false accepts are the dominant Mandarin KWS error: 打开 (dǎ-kāi) vs 大开
(dà-kāi) differ only in a tone, and a weak model fires on the wrong one. We generate a
keyword's acoustically-confusable neighbours by single edits to its tonal pinyin:

* **tone swaps** — Tone 2↔3 (the classic confusion), and any tone ↔ neutral (5);
* **initial confusions** — zh/z, ch/c, sh/s, n/l, f/h;
* **final confusions** — front/back nasals in/ing, en/eng, an/ang.

Each confusable is returned as ``(pinyin, unit_ids)`` so it can be scored by the same keyword
spotter as a *negative*. Used to (a) measure a model's confusion margin — target score minus
best-confusable score, positive when the model separates them — and (b) build a hard-negative
slice for FRR@FAH. The ``(label, unit_ids)`` shape is exactly what an LLM generator of natural
confusable *words* would also return, so an LLM source drops in behind the same interface.
"""

from __future__ import annotations

from .tokenizer import PinyinTokenizer
from .tokenizer.pinyin import split_syllable, syllable_to_units, text_to_tone3

# Most-confusable tone substitutions per tone (1..5; 5 = neutral).
_TONE_CONFUSIONS = {
    "1": ["5"],
    "2": ["3", "5"],
    "3": ["2", "5"],
    "4": ["5"],
    "5": ["2", "3"],
}
_INITIAL_PAIRS = {
    "zh": "z", "z": "zh", "ch": "c", "c": "ch", "sh": "s", "s": "sh",
    "n": "l", "l": "n", "f": "h", "h": "f",
}
_FINAL_PAIRS = {
    "in": "ing", "ing": "in", "en": "eng", "eng": "en", "an": "ang", "ang": "an",
}

Part = tuple[str, str, str]  # (initial, final, tone)
Confusable = tuple[str, list[int]]  # (pinyin label, unit ids)


def _parts(text: str) -> list[Part]:
    return [split_syllable(s) for s in text_to_tone3(text)]


def _parts_to_units(parts: list[Part], tok: PinyinTokenizer) -> list[str]:
    units: list[str] = []
    for ini, fin, tone in parts:
        units.extend(syllable_to_units(f"{ini}{fin}{tone}", tok.mode))
    return units


def _pinyin(parts: list[Part]) -> str:
    return " ".join(f"{i}{f}{t}" for i, f, t in parts)


def tone_confusables(text: str, tokenizer: PinyinTokenizer, max_variants: int = 12) -> list[Confusable]:
    """Single-edit acoustically-confusable variants of ``text`` (tone/initial/final swaps).

    Excludes the original and any variant that is not fully in-vocab. Order: tone swaps first
    (highest-value), then initial, then final confusions.
    """
    parts = _parts(text)
    if not parts:
        return []
    original = tuple(tokenizer.encode(text))
    seen: set[tuple[int, ...]] = {original}
    out: list[Confusable] = []
    unk = tokenizer.unk_id

    def edits():
        for i, (ini, fin, tone) in enumerate(parts):
            for nt in _TONE_CONFUSIONS.get(tone, []):
                yield [*parts[:i], (ini, fin, nt), *parts[i + 1:]]
        for i, (ini, fin, tone) in enumerate(parts):
            if ini in _INITIAL_PAIRS:
                yield [*parts[:i], (_INITIAL_PAIRS[ini], fin, tone), *parts[i + 1:]]
        for i, (ini, fin, tone) in enumerate(parts):
            if fin in _FINAL_PAIRS:
                yield [*parts[:i], (ini, _FINAL_PAIRS[fin], tone), *parts[i + 1:]]

    for cand in edits():
        units = _parts_to_units(cand, tokenizer)
        ids = [tokenizer.unit_to_id.get(u, unk) for u in units]
        key = tuple(ids)
        if unk in ids or key in seen:
            continue
        seen.add(key)
        out.append((_pinyin(cand), ids))
        if len(out) >= max_variants:
            break
    return out


def confusion_margin(log_probs, text: str, tokenizer: PinyinTokenizer, spotter,
                     max_variants: int = 12) -> float | None:
    """Target detection score minus the best tone-confusable score for one utterance.

    Positive means the model ranks the true keyword above every confusable (good false-accept
    behaviour). Returns None if no confusables could be generated.
    """
    confs = tone_confusables(text, tokenizer, max_variants)
    if not confs:
        return None
    target = spotter.score(log_probs, tokenizer.encode(text))
    best_conf = max(spotter.score(log_probs, ids) for _, ids in confs)
    return target - best_conf
