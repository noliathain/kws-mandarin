"""Build train/dev/test manifests from an extracted AISHELL-1 corpus.

AISHELL-1 layout after extraction (see ``scripts/download_data.sh``)::

    <aishell_root>/
        transcript/aishell_transcript_v0.8.txt      # "<utt_id> 甲 醛 白 血 病 ..."
        wav/{train,dev,test}/<speaker>/<utt_id>.wav  # 16 kHz mono

Durations are read from the wav header only (stdlib ``wave`` — no audio decode, no torch),
parallelized across CPUs. Run:

    uv run python -m kws_mandarin.data.prepare_aishell \\
        --aishell-root <root>/corpora/aishell1/data_aishell \\
        --out          <root>/manifests \\
        --workers      $(nproc)
"""

from __future__ import annotations

import argparse
import contextlib
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .manifest import Utterance, manifest_stats, write_manifest

SPLITS = ("train", "dev", "test")
_TRANSCRIPT = "transcript/aishell_transcript_v0.8.txt"


def wav_duration(path: str) -> float:
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getnframes() / float(w.getframerate())


def load_transcripts(aishell_root: Path) -> dict[str, str]:
    """utt_id -> Han text (separators stripped)."""
    path = aishell_root / _TRANSCRIPT
    if not path.exists():
        raise FileNotFoundError(f"transcript not found: {path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            utt_id, tokens = parts[0], parts[1:]
            out[utt_id] = "".join(tokens)
    return out


def _duration_task(rec: tuple[str, str, str, str, str]) -> dict | None:
    utt_id, wav, text, speaker, split = rec
    try:
        dur = wav_duration(wav)
    except Exception:
        return None  # unreadable / truncated wav — dropped and counted by caller
    return {
        "utt_id": utt_id,
        "wav": wav,
        "text": text,
        "duration": round(dur, 3),
        "speaker": speaker,
        "split": split,
    }


def _collect_wavs(aishell_root: Path, split: str) -> list[Path]:
    split_dir = aishell_root / "wav" / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*/*.wav"))


def build_manifests(
    aishell_root: str | Path,
    out_dir: str | Path,
    workers: int = 8,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, dict]:
    aishell_root = Path(aishell_root)
    out_dir = Path(out_dir)
    transcripts = load_transcripts(aishell_root)

    report: dict[str, dict] = {}
    for split in splits:
        wavs = _collect_wavs(aishell_root, split)
        tasks: list[tuple[str, str, str, str, str]] = []
        missing_text = 0
        for wav in wavs:
            utt_id = wav.stem
            text = transcripts.get(utt_id)
            if text is None:
                missing_text += 1
                continue
            speaker = wav.parent.name
            tasks.append((utt_id, str(wav.resolve()), text, speaker, split))

        if workers > 1 and tasks:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_duration_task, tasks, chunksize=256))
        else:
            results = [_duration_task(t) for t in tasks]

        utts = [Utterance(**r) for r in results if r is not None]
        unreadable = len(tasks) - len(utts)
        out_path = out_dir / f"aishell1_{split}.jsonl"
        write_manifest(out_path, utts)

        stats = manifest_stats(utts)
        stats.update(
            path=str(out_path),
            wavs_found=len(wavs),
            missing_transcript=missing_text,
            unreadable_wav=unreadable,
        )
        report[split] = stats
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Build AISHELL-1 manifests")
    ap.add_argument("--aishell-root", required=True, help="path to extracted data_aishell/")
    ap.add_argument("--out", required=True, help="output directory for manifests")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = ap.parse_args()

    report = build_manifests(args.aishell_root, args.out, args.workers, tuple(args.splits))
    print(f"{'split':>6} {'utts':>8} {'hours':>8} {'spk':>5}  notes")
    for split, s in report.items():
        note = []
        if s.get("missing_transcript"):
            note.append(f"{s['missing_transcript']} no-transcript")
        if s.get("unreadable_wav"):
            note.append(f"{s['unreadable_wav']} unreadable")
        print(
            f"{split:>6} {s['utterances']:>8} {s['hours']:>8} {s.get('speakers', 0):>5}  "
            f"{'; '.join(note)}  -> {s['path']}"
        )


if __name__ == "__main__":
    main()
