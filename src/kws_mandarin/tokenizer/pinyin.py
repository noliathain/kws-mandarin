"""Tonal-pinyin tokenizer for open-vocabulary Mandarin KWS.

A keyword in an open-vocabulary system is a *token sequence resolved at runtime*, so the
unit inventory must be (a) fixed and complete — every id the CTC head can emit is known up
front — and (b) compositional — any Chinese string maps to a sequence of in-vocab units.

We build the inventory from tonal pinyin. Three granularities are supported:

- ``ToneMode.FINAL``    initial + toned-final, e.g. ``中``  -> ``["zh", "ong1"]``  (~200 units)
- ``ToneMode.SEPARATE`` initial + final + tone,  e.g. ``中``  -> ``["zh", "ong", "1"]`` (~65 units)
- ``ToneMode.SYLLABLE`` whole toned syllable,    e.g. ``中``  -> ``["zhong1"]``          (~1300 units)

``FINAL`` is the default: it is the sweet spot between the tiny ``SEPARATE`` inventory
(most data-efficient, but tone becomes a discrete token detached from its host final) and
the large ``SYLLABLE`` inventory (matches the CRNN-CTC precedent, but a big softmax).

The vocabulary is generated deterministically from a character sweep (see
``scripts/build_vocab.py``) and committed as a text file per mode, so it is versioned,
inspectable, and identical across machines. Splitting a toned pinyin syllable into
initial/final/tone is a pure string operation, which *preserves pypinyin's phrase-context
disambiguation of heteronyms* (多音字) — we never re-run pinyin per character.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pypinyin import Style, pinyin

# Two- and one-letter Mandarin initials (声母). Everything else at syllable start
# (y, w, or a vowel) is a zero-initial syllable and the glide stays in the final.
_INITIALS_2 = frozenset({"zh", "ch", "sh"})
_INITIALS_1 = frozenset("bpmfdtnlgkhjqxrzcs")

_VOCAB_DIR = Path(__file__).parent / "vocab"


class ToneMode(str, Enum):
    FINAL = "final"
    SEPARATE = "separate"
    SYLLABLE = "syllable"


def split_syllable(syllable_tone3: str) -> tuple[str, str, str]:
    """Split a TONE3 pinyin syllable (e.g. ``"zhong1"``) into (initial, final, tone).

    Zero-initial syllables return ``initial == ""``. Tone is ``"1".."5"`` where ``5`` is
    the neutral tone. Input is assumed to already carry a trailing tone digit.
    """
    syl = syllable_tone3
    if syl and syl[-1].isdigit():
        tone, body = syl[-1], syl[:-1]
    else:
        tone, body = "5", syl
    if body[:2] in _INITIALS_2:
        return body[:2], body[2:], tone
    if body[:1] in _INITIALS_1:
        return body[:1], body[1:], tone
    return "", body, tone


def text_to_tone3(text: str) -> list[str]:
    """Convert Han text to a flat list of TONE3 syllables, dropping non-Han tokens.

    Phrase context is used for heteronym disambiguation (pypinyin default).
    """
    rows = pinyin(
        text,
        style=Style.TONE3,
        strict=True,
        neutral_tone_with_five=True,
        errors="ignore",  # drop anything without a Han reading
    )
    out: list[str] = []
    for row in rows:
        syl = row[0]
        # Keep only tokens that came out as a toned pinyin syllable.
        if syl and syl[-1].isdigit():
            out.append(syl)
    return out


def syllable_to_units(syllable_tone3: str, mode: ToneMode) -> list[str]:
    """Expand one TONE3 syllable into unit tokens for the given granularity."""
    if mode is ToneMode.SYLLABLE:
        return [syllable_tone3]
    initial, final, tone = split_syllable(syllable_tone3)
    if mode is ToneMode.FINAL:
        units = []
        if initial:
            units.append(initial)
        units.append(f"{final}{tone}")  # toned final, e.g. "ong1"
        return units
    if mode is ToneMode.SEPARATE:
        units = []
        if initial:
            units.append(initial)
        units.append(final)
        units.append(tone)
        return units
    raise ValueError(f"unknown mode: {mode!r}")


class PinyinTokenizer:
    """Maps Han text <-> a fixed inventory of tonal-pinyin units for a CTC head.

    Id 0 is always the CTC blank; id 1 is ``<unk>`` (a defensive fallback that should
    never fire for in-domain Han text). Real units follow, in the order given by the
    committed vocab file for the mode.
    """

    BLANK = "<blank>"
    UNK = "<unk>"

    def __init__(self, mode: ToneMode = ToneMode.FINAL, units: list[str] | None = None):
        self.mode = ToneMode(mode)
        if units is None:
            units = self._load_units(self.mode)
        # Special tokens first; blank MUST be id 0 for torch's CTCLoss default.
        self.id_to_unit: list[str] = [self.BLANK, self.UNK, *units]
        self.unit_to_id: dict[str, int] = {u: i for i, u in enumerate(self.id_to_unit)}

    # -- construction ------------------------------------------------------------------

    @staticmethod
    def _load_units(mode: ToneMode) -> list[str]:
        path = _VOCAB_DIR / f"{mode.value}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"vocab file missing: {path}. Generate it with "
                f"`uv run python scripts/build_vocab.py`."
            )
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # -- encoding ----------------------------------------------------------------------

    def units_for_text(self, text: str) -> list[str]:
        out: list[str] = []
        for syl in text_to_tone3(text):
            out.extend(syllable_to_units(syl, self.mode))
        return out

    def encode(self, text: str) -> list[int]:
        unk = self.unit_to_id[self.UNK]
        return [self.unit_to_id.get(u, unk) for u in self.units_for_text(text)]

    def decode_units(self, ids: list[int]) -> list[str]:
        return [self.id_to_unit[i] for i in ids]

    # -- properties --------------------------------------------------------------------

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_unit)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"PinyinTokenizer(mode={self.mode.value!r}, vocab_size={self.vocab_size})"
