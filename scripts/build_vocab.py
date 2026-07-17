"""Generate the committed unit-vocabulary files for every tone mode.

Sweeps the common CJK ideograph range through pypinyin, splits each reading into units
with the exact same code path the tokenizer uses at runtime, and writes one sorted unit
per line to ``src/kws_mandarin/tokenizer/vocab/<mode>.txt``.

Deterministic and idempotent. Run after any change to the split logic:

    uv run python scripts/build_vocab.py
"""

from __future__ import annotations

from pathlib import Path

from pypinyin import Style, pinyin

from kws_mandarin.tokenizer.pinyin import ToneMode, syllable_to_units

_VOCAB_DIR = Path(__file__).resolve().parents[1] / "src" / "kws_mandarin" / "tokenizer" / "vocab"

# CJK Unified Ideographs (basic block) covers all standard modern readings.
_CJK_START, _CJK_END = 0x4E00, 0x9FFF


def all_readings() -> set[str]:
    readings: set[str] = set()
    for cp in range(_CJK_START, _CJK_END + 1):
        rows = pinyin(
            chr(cp),
            style=Style.TONE3,
            strict=True,
            neutral_tone_with_five=True,
            errors="ignore",
        )
        for row in rows:
            syl = row[0]
            if syl and syl[-1].isdigit():
                readings.add(syl)
    return readings


def main() -> None:
    _VOCAB_DIR.mkdir(parents=True, exist_ok=True)
    readings = all_readings()
    print(f"collected {len(readings)} distinct toned syllables")
    for mode in ToneMode:
        units: set[str] = set()
        for syl in readings:
            units.update(syllable_to_units(syl, mode))
        ordered = sorted(units)
        out = _VOCAB_DIR / f"{mode.value}.txt"
        out.write_text("\n".join(ordered) + "\n", encoding="utf-8")
        print(f"{mode.value:8s} -> {len(ordered):4d} units  ({out})")


if __name__ == "__main__":
    main()
