"""Corpus data-quality validation — fail loud (CLAUDE.md Rule 12).

Runs after manifests are built and *before* training, to catch problems that would otherwise
degrade silently:

Hard failures (FAIL — must be fixed, CLI exits non-zero):
  * ``ctc_feasibility``   — an utterance with fewer encoder frames than its CTC target needs
                            produces inf loss and is silently dropped by ``zero_infinity``.
  * ``speaker_leakage``   — a speaker appearing in more than one split invalidates evaluation.
  * ``empty_target``      — text that tokenizes to zero units (nothing to align).
  * ``duplicate_utt_id``  — the same id in multiple utterances.
  * ``missing_wav``       — a referenced wav no longer exists.
  * ``audio_format``      — (deep) wrong sample rate or non-mono audio.

Soft warnings (WARN — surfaced, not fatal):
  * ``has_unk``           — target contains the <unk> fallback (tokenizer coverage gap).
  * ``duration_range``    — clips shorter/longer than expected bounds.
  * ``duration_mismatch`` — (deep) manifest duration disagrees with the wav header.

The default pass is manifest-only and fast (CTC feasibility is derived from manifest
duration). ``--deep`` additionally reads each wav *header* (soundfile.info, no decode) to
verify sample rate, channels, and true duration.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..tokenizer import PinyinTokenizer, ToneMode
from .manifest import read_manifest

_EXPECTED_SR = 16000
_HOP = 160  # must match LogMelFrontend.hop_length

_TOK_CACHE: dict[str, PinyinTokenizer] = {}


def _tok(mode: str) -> PinyinTokenizer:
    if mode not in _TOK_CACHE:
        _TOK_CACHE[mode] = PinyinTokenizer(ToneMode(mode))
    return _TOK_CACHE[mode]


def _analyze(args: tuple) -> dict:
    rec, mode, sample_rate, hop, deep = args
    tok = _tok(mode)
    ids = tok.encode(rec["text"])
    u = len(ids)
    # exact CTC minimum: one frame per label + one blank between each adjacent duplicate pair
    required = u + sum(1 for i in range(1, u) if ids[i] == ids[i - 1])
    samples = int(rec["duration"] * sample_rate)
    enc_frames = 1 + samples // hop

    out = {
        "utt_id": rec["utt_id"],
        "speaker": rec["speaker"],
        "split": rec["split"],
        "duration": rec["duration"],
        "n_units": u,
        "has_unk": tok.unk_id in ids,
        "required_frames": required,
        "enc_frames": enc_frames,
        "feasible": u > 0 and enc_frames >= required,
        "exists": os.path.exists(rec["wav"]),
        "sr": None,
        "channels": None,
        "actual_dur": None,
    }
    if deep and out["exists"]:
        try:
            import soundfile as sf

            info = sf.info(rec["wav"])
            out["sr"] = info.samplerate
            out["channels"] = info.channels
            out["actual_dur"] = info.frames / info.samplerate if info.samplerate else None
        except Exception:
            out["sr"] = -1  # sentinel: unreadable header
    return out


@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str
    examples: list = field(default_factory=list)


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str, examples: list | None = None) -> None:
        self.checks.append(Check(name, status, detail, list(examples or [])[:5]))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def render(self) -> str:
        icon = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}
        lines = []
        for c in self.checks:
            lines.append(f"[{icon[c.status]}] {c.name}: {c.detail}")
            for ex in c.examples:
                lines.append(f"          e.g. {ex}")
        n_fail = len(self.failures)
        n_warn = sum(1 for c in self.checks if c.status == "WARN")
        verdict = "PASS" if self.ok else "FAIL"
        lines.append(f"\nRESULT: {verdict}  ({n_fail} failure(s), {n_warn} warning(s))")
        return "\n".join(lines)


def validate_manifests(
    manifest_paths: dict[str, str],
    mode: str = "final",
    sample_rate: int = _EXPECTED_SR,
    hop_length: int = _HOP,
    dur_min: float = 0.3,
    dur_max: float = 20.0,
    deep: bool = False,
    workers: int = 8,
) -> ValidationReport:
    tasks = []
    for split, path in manifest_paths.items():
        for u in read_manifest(path):
            rec = asdict(u)
            rec["split"] = split
            tasks.append((rec, mode, sample_rate, hop_length, deep))

    if workers > 1 and tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_analyze, tasks, chunksize=512))
    else:
        rows = [_analyze(t) for t in tasks]

    report = ValidationReport()
    n = len(rows)
    report.summary = {"utterances": n}
    if n == 0:
        report.add("empty_corpus", "FAIL", "no utterances found in any manifest")
        return report

    # --- hard failures ------------------------------------------------------------------
    missing = [r["utt_id"] for r in rows if not r["exists"]]
    report.add("missing_wav", "FAIL" if missing else "PASS",
               f"{len(missing)} referenced wav(s) do not exist" if missing
               else "all referenced wavs exist", missing)

    empty = [r["utt_id"] for r in rows if r["n_units"] == 0]
    report.add("empty_target", "FAIL" if empty else "PASS",
               f"{len(empty)} utt(s) tokenize to zero units" if empty
               else "all utts have >=1 unit", empty)

    infeasible = [r for r in rows if r["n_units"] > 0 and not r["feasible"]]
    ex = [f"{r['utt_id']} ({r['enc_frames']} frames < {r['required_frames']} needed)"
          for r in infeasible]
    report.add("ctc_feasibility", "FAIL" if infeasible else "PASS",
               f"{len(infeasible)} utt(s) too short for their CTC target (would be dropped)"
               if infeasible else "all utts have enough frames for CTC", ex)

    dup = [uid for uid, c in Counter(r["utt_id"] for r in rows).items() if c > 1]
    report.add("duplicate_utt_id", "FAIL" if dup else "PASS",
               f"{len(dup)} duplicated utt_id(s)" if dup else "all utt_ids unique", dup)

    spk_splits: dict[str, set] = defaultdict(set)
    for r in rows:
        spk_splits[r["speaker"]].add(r["split"])
    leaked = [f"{s} in {sorted(v)}" for s, v in spk_splits.items() if len(v) > 1]
    report.add("speaker_leakage", "FAIL" if leaked else "PASS",
               f"{len(leaked)} speaker(s) span multiple splits" if leaked
               else "speakers are disjoint across splits", leaked)

    # --- soft warnings ------------------------------------------------------------------
    unk = [r["utt_id"] for r in rows if r["has_unk"]]
    report.add("has_unk", "WARN" if unk else "PASS",
               f"{len(unk)} utt(s) contain <unk>" if unk else "no <unk> in any target", unk)

    outliers = [f"{r['utt_id']} ({r['duration']}s)" for r in rows
                if r["duration"] < dur_min or r["duration"] > dur_max]
    report.add("duration_range", "WARN" if outliers else "PASS",
               f"{len(outliers)} clip(s) outside [{dur_min}, {dur_max}]s" if outliers
               else f"all clips within [{dur_min}, {dur_max}]s", outliers)

    # --- deep (audio header) checks -----------------------------------------------------
    if deep:
        bad_fmt = [f"{r['utt_id']} (sr={r['sr']}, ch={r['channels']})" for r in rows
                   if r["exists"] and (r["sr"] not in (sample_rate,) or r["channels"] != 1)]
        report.add("audio_format", "FAIL" if bad_fmt else "PASS",
                   f"{len(bad_fmt)} wav(s) not {sample_rate} Hz mono" if bad_fmt
                   else f"all wavs are {sample_rate} Hz mono", bad_fmt)

        mism = [f"{r['utt_id']} (manifest {r['duration']}s vs header {r['actual_dur']:.2f}s)"
                for r in rows if r["actual_dur"] is not None
                and abs(r["actual_dur"] - r["duration"]) > 0.05]
        report.add("duration_mismatch", "WARN" if mism else "PASS",
                   f"{len(mism)} duration mismatch(es) > 50 ms" if mism
                   else "manifest durations match wav headers", mism)

    # --- summary stats ------------------------------------------------------------------
    per_split = defaultdict(lambda: {"utts": 0, "hours": 0.0, "speakers": set()})
    for r in rows:
        s = per_split[r["split"]]
        s["utts"] += 1
        s["hours"] += r["duration"] / 3600.0
        s["speakers"].add(r["speaker"])
    report.summary["splits"] = {
        k: {"utts": v["utts"], "hours": round(v["hours"], 2), "speakers": len(v["speakers"])}
        for k, v in per_split.items()
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate KWS corpus manifests (fail loud)")
    ap.add_argument("--manifests", required=True, help="dir containing aishell1_<split>.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--mode", default="final", choices=[m.value for m in ToneMode])
    ap.add_argument("--deep", action="store_true", help="also read wav headers (sr/channels)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    mdir = Path(args.manifests)
    paths = {s: str(mdir / f"aishell1_{s}.jsonl") for s in args.splits
             if (mdir / f"aishell1_{s}.jsonl").exists()}
    report = validate_manifests(paths, mode=args.mode, deep=args.deep, workers=args.workers)
    print(report.render())
    print("\nsummary:", report.summary)
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
