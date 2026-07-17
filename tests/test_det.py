import numpy as np
import pytest

from kws_mandarin.eval import det_curve, frr_at_fah, summary


def test_perfect_separation():
    # positives all above negatives -> a threshold exists with FRR=0 and FAH=0.
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    durations = [900] * 8  # negatives sum to 4*900s = 1.0 hour
    op = frr_at_fah(scores, labels, durations, target_fah=0.0)
    assert op.frr == 0.0
    assert op.fah == 0.0


def test_overlap_hand_computed():
    # 3 positives, 3 negatives, negative audio = 3*1200s = 1.0 hour.
    scores = [0.9, 0.8, 0.3, 0.7, 0.4, 0.2]
    labels = [1, 1, 1, 0, 0, 0]
    durations = [1200] * 6

    # FAH must be 0: only threshold above 0.7 works; positive 0.3 is missed -> FRR = 1/3.
    op0 = frr_at_fah(scores, labels, durations, target_fah=0.0)
    assert op0.frr == pytest.approx(1 / 3)
    assert op0.fah == 0.0

    # Allow up to 1 FA/hr: can admit the 0.7 negative, but 0.3 positive still needs a
    # threshold that also admits two negatives (FAH=2). So FRR floor stays 1/3.
    op1 = frr_at_fah(scores, labels, durations, target_fah=1.0)
    assert op1.frr == pytest.approx(1 / 3)
    assert op1.fah <= 1.0


def test_det_curve_monotonic_and_spans_range():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(1.0, 1.0, 200), rng.normal(-1.0, 1.0, 800)])
    labels = np.array([1] * 200 + [0] * 800)
    durations = np.full(1000, 3.0)
    thr, frr, fah = det_curve(scores, labels, durations)

    assert np.all(np.diff(thr) > 0)            # thresholds ascending
    assert np.all(np.diff(frr) >= -1e-12)      # FRR non-decreasing in threshold
    assert np.all(np.diff(fah) <= 1e-12)       # FAH non-increasing in threshold
    assert frr[0] == 0.0 and fah[-1] == 0.0    # extremes: fire-all / fire-nothing


def test_looser_budget_never_worsens_frr():
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(1.0, 1.0, 100), rng.normal(-1.0, 1.0, 400)])
    labels = np.array([1] * 100 + [0] * 400)
    durations = np.full(500, 4.0)
    frr_tight = frr_at_fah(scores, labels, durations, 0.5).frr
    frr_loose = frr_at_fah(scores, labels, durations, 5.0).frr
    assert frr_loose <= frr_tight


def test_summary_keys():
    scores = [0.9, 0.1, 0.8, 0.2]
    labels = [1, 0, 1, 0]
    durations = [1800, 1800, 1800, 1800]
    out = summary(scores, labels, durations, target_fahs=(0.5, 1.0))
    assert set(out) == {0.5, 1.0}
