"""KWS evaluation: false-reject rate at a fixed false-alarms-per-hour (FRR @ FAH).

There is no single "accuracy" number that means anything for keyword spotting. A KWS system
is defined by its Detection-Error-Tradeoff curve: how often it *misses* a real keyword
(false-reject rate) at a given rate of *spurious* triggers (false alarms per hour of
non-keyword audio). Every modeling decision in this repo — units, wildcard arcs,
augmentation SNR — is adjudicated by moving this curve, not by a loss value.

Conventions
-----------
- ``scores``    : detection score per trial (higher = more keyword-like).
- ``labels``    : bool/int, 1 if the keyword is truly present in the trial, else 0.
- ``durations`` : trial length in seconds. Only the durations of *negative* trials enter
                  the FAH denominator (false alarms are counted over non-keyword audio).
- A trial fires when ``score >= threshold``.
    * positive trial, does not fire  -> false reject
    * negative trial, fires          -> false alarm
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    frr: float  # false-reject rate in [0, 1]
    fah: float  # false alarms per hour


def _as_arrays(scores, labels, durations):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    durations = np.asarray(durations, dtype=np.float64)
    if not (len(scores) == len(labels) == len(durations)):
        raise ValueError("scores, labels, durations must be the same length")
    if len(scores) == 0:
        raise ValueError("no trials provided")
    return scores, labels, durations


def det_curve(scores, labels, durations):
    """Return ``(thresholds, frr, fah)`` arrays swept over candidate thresholds.

    Arrays are aligned and sorted by ascending threshold. The sweep includes a threshold
    above every score (fires nothing: FRR=1, FAH=0) and one below every score (fires
    everything: FRR=0, FAH maximal), so the full operating range is represented.
    """
    scores, labels, durations = _as_arrays(scores, labels, durations)

    pos = np.sort(scores[labels])
    neg_scores = scores[~labels]
    neg_hours = durations[~labels].sum() / 3600.0
    if neg_hours <= 0:
        raise ValueError("total duration of negative trials must be > 0 to compute FAH")
    neg = np.sort(neg_scores)
    n_pos = len(pos)
    if n_pos == 0:
        raise ValueError("need at least one positive trial to compute FRR")

    # Candidate thresholds: just above/below each distinct score. Using midpoints between
    # sorted unique scores (plus outer sentinels) makes ">=" ties unambiguous.
    uniq = np.unique(scores)
    lo, hi = uniq[0] - 1.0, uniq[-1] + 1.0
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    thresholds = np.concatenate(([lo], mids, [hi]))

    # FRR(t) = fraction of positives strictly below t = searchsorted(pos, t, 'left') / n_pos
    frr = np.searchsorted(pos, thresholds, side="left") / n_pos
    # FA count(t) = negatives with score >= t = len(neg) - searchsorted(neg, t, 'left')
    fa_count = len(neg) - np.searchsorted(neg, thresholds, side="left")
    fah = fa_count / neg_hours
    return thresholds, frr, fah


def frr_at_fah(scores, labels, durations, target_fah: float) -> OperatingPoint:
    """Operating point with the lowest FRR while keeping FAH <= ``target_fah``.

    Lowering the threshold trades FRR down for FAH up, so the answer is the most permissive
    threshold that still satisfies the FAH budget. A feasible point always exists (the
    highest threshold fires nothing: FAH=0).
    """
    thresholds, frr, fah = det_curve(scores, labels, durations)
    feasible = np.where(fah <= target_fah)[0]
    if len(feasible) == 0:  # only possible if target_fah < 0
        raise ValueError(f"no operating point achieves FAH <= {target_fah}")
    best = feasible[np.argmin(frr[feasible])]
    return OperatingPoint(
        threshold=float(thresholds[best]),
        frr=float(frr[best]),
        fah=float(fah[best]),
    )


def summary(scores, labels, durations, target_fahs=(0.5, 1.0, 2.0)) -> dict[float, OperatingPoint]:
    """FRR at a set of FAH operating points — the standard KWS report line."""
    return {t: frr_at_fah(scores, labels, durations, t) for t in target_fahs}
