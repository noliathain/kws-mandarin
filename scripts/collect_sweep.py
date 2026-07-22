"""Collect validation curves from training logs into one JSON, keyed by model scale.

Each run writes its own timestamped log whose header records the config it used, so the
scale is recoverable without any bookkeeping on the side. Used to plot the capacity sweep.

    uv run python scripts/collect_sweep.py --out /tmp/sweep.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import yaml

LOG_DIR = "/teamspace/lightning_storage/kws-mandarin/logs"
VAL = re.compile(r"\[val\] step (\d+) ter=([\d.]+)(?: frr@0\.5=([\d.]+))?(?: frr@1\.0=([\d.]+))?")
START = re.compile(r">> training start \S+ nproc=(\d+) config=(\S+)")
PARAMS = re.compile(r"model params: ([\d,]+)")
# param count -> scale, so results survive their config file being deleted
_SCALE_BY_PARAMS = {62260: 3.0, 189880: 6.0, 642112: 12.0, 1653664: 20.0, 3574744: 30.0, 6225424: 40.0}
STEP = re.compile(r"^step (\d+)/\d+ loss ([\d.]+) lr \S+ ([\d.]+) it/s", re.M)


def parse(path: str) -> dict | None:
    text = Path(path).read_text(errors="ignore")
    m = START.search(text)
    if not m:
        return None
    cfg_path = m.group(2)
    params = int(PARAMS.search(text).group(1).replace(",", "")) if PARAMS.search(text) else None
    if Path(cfg_path).exists():
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        scale, micro = cfg["model"]["scale"], cfg["data"]["batch_size"]
        accum = cfg["optim"].get("accum_steps", 1)
    elif params is not None:
        # A config deleted after its run must not erase the result: recover the scale from the
        # parameter count the log records, and mark the batch geometry unknown.
        scale, micro, accum = _SCALE_BY_PARAMS.get(params, -1.0), None, None
    else:
        return None
    vals = [
        {"step": int(s), "ter": float(t),
         "frr05": float(f5) if f5 else None, "frr10": float(f10) if f10 else None}
        for s, t, f5, f10 in VAL.findall(text)
    ]
    if not vals:
        return None
    steps = [(int(s), float(l), float(r)) for s, l, r in STEP.findall(text)]
    return {
        "log": path,
        "config": cfg_path,
        "scale": scale,
        "params": params,
        "batch_per_gpu": micro,
        "accum_steps": accum,
        "nproc": int(m.group(1)),
        "val": vals,
        "loss": [{"step": s, "loss": l} for s, l, _ in steps],
        "it_s_median": sorted(r for _, _, r in steps)[len(steps) // 2] if steps else None,
        "finished": ">> training exit 0" in text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default=LOG_DIR)
    args = ap.parse_args()

    runs = [r for r in (parse(p) for p in sorted(glob.glob(f"{args.log_dir}/train_*.log"))) if r]
    # one run per scale: keep the most recent complete curve for each
    best: dict[float, dict] = {}
    for r in runs:
        prev = best.get(r["scale"])
        if prev is None or len(r["val"]) >= len(prev["val"]):
            best[r["scale"]] = r
    out = [best[k] for k in sorted(best)]
    Path(args.out).write_text(json.dumps(out, indent=2))
    for r in out:
        last = r["val"][-1]
        geom = (f"micro {r['batch_per_gpu']}x{r['accum_steps']}"
                if r["batch_per_gpu"] else "micro ?")
        prm = f"{r['params']:,}" if r["params"] else "?"
        print(f"  scale {r['scale']:5.1f} ({prm:>9s}p)  {geom:12s}  step {last['step']:5d}  "
              f"ter={last['ter']:.4f}  frr@1.0={last['frr10']}  "
              f"{'done' if r['finished'] else 'incomplete'}")


if __name__ == "__main__":
    main()
