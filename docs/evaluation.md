# Evaluation protocol — read before trusting any number

KWS has **no meaningful accuracy scalar.** A detector is characterised by the trade-off
between two errors:

- **False reject (miss):** a real keyword utterance fails to fire. Reported as
  **FRR** (false-reject rate), the fraction of positive trials missed.
- **False alarm:** the detector fires on non-keyword audio. Reported as **FAH**
  (false alarms per hour), counted over the duration of *negative* audio.

A trial fires when `score >= threshold`. Sweeping the threshold traces the **DET curve**.
The headline number is always **FRR @ a fixed FAH** (e.g. "3.1% FRR @ 0.5 FA/hr"), and it
must be reported **per SNR band** — a system that looks great at 20 dB can collapse at 0 dB.

## Using the harness

```python
from kws_mandarin.eval import det_curve, frr_at_fah, summary

# scores: detector score per trial (higher = more keyword-like)
# labels: 1 if keyword truly present, else 0
# durations: trial length in seconds (only negatives count toward the FAH denominator)

op = frr_at_fah(scores, labels, durations, target_fah=0.5)
print(op.frr, op.threshold)

report = summary(scores, labels, durations, target_fahs=(0.5, 1.0, 2.0))
```

`frr_at_fah` returns the lowest FRR achievable while keeping FAH within budget (the most
permissive threshold that still respects the FAH ceiling). `det_curve` returns the full
`(thresholds, frr, fah)` sweep for plotting.

## Rules of the road

- **Report per SNR.** Bucket trials by SNR band and give a DET curve for each. Never report
  a single pooled number.
- **Negatives must be realistic.** False alarms are what kill a shipped KWS system. Draw
  negatives from continuous non-keyword speech and noise, not silence.
- **Confusable negatives get their own slice.** Tone-confusable near-homophones are the
  hard case; track FRR/FAH on that slice separately so tone errors can't hide in the pool.
- **Fixed operating points.** Pick FAH budgets up front (0.5 / 1.0 / 2.0 FA/hr) and compare
  systems at those points, not at whatever threshold flatters each one.
