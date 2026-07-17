"""Corpus-agnostic manifest format.

A manifest is a JSONL file, one utterance per line. It is deliberately **tokenizer-agnostic**
— it stores the raw Han ``text`` and lets the training Dataset tokenize on the fly for
whatever unit mode is configured, so we never re-prepare data to try a different unit
inventory. Audio is referenced by absolute path; corpora are never copied into the repo.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class Utterance:
    utt_id: str
    wav: str          # absolute path to a 16 kHz mono wav
    text: str         # Han transcript, no separators
    duration: float   # seconds
    speaker: str
    split: str        # "train" | "dev" | "test"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Utterance":
        return cls(**json.loads(line))


def write_manifest(path: str | Path, utterances: Iterable[Utterance]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for u in utterances:
            f.write(u.to_json() + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> list[Utterance]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [Utterance.from_json(line) for line in f if line.strip()]


def manifest_stats(utterances: list[Utterance]) -> dict:
    if not utterances:
        return {"utterances": 0, "hours": 0.0, "speakers": 0}
    total_s = sum(u.duration for u in utterances)
    return {
        "utterances": len(utterances),
        "hours": round(total_s / 3600.0, 2),
        "speakers": len({u.speaker for u in utterances}),
        "avg_dur_s": round(total_s / len(utterances), 2),
    }
