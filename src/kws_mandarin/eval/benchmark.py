"""Open-vocab KWS benchmark construction: keyword selection and LibriPhrase-style trials.

Turns a held-out manifest into a large, curated trial list so the model can be scored with
EER / AUC / recall@FAH — the metrics the open-vocab KWS literature reports. Following
LibriPhrase, each keyword gets positive trials and two kinds of negative:

  * Easy  — a random keyword the utterance does NOT contain (different sounds entirely).
  * Hard  — a tone/initial/final-confusable variant of a keyword the utterance DOES contain,
            but which was not actually said. The audio sounds close to the query, so only a
            model that resolves the fine phonetic distinction rejects it. This is the Mandarin
            analogue of LibriPhrase's phonetic "swap" negatives.

These are pure functions over text — no model — so the trial list is deterministic and unit
tested. The model scores the trials in scripts/benchmark_kws.py.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from ..confusables import tone_confusables
from ..data.manifest import Utterance
from ..tokenizer import PinyinTokenizer


@dataclass(frozen=True)
class Trial:
    utt_id: str
    query: str            # the keyword text being tested against this utterance
    query_ids: tuple[int, ...]
    label: int            # 1 if the utterance truly contains `query`, else 0
    split: str            # "pos" | "easy" | "hard"
    duration: float


def select_keywords(
    utts: list[Utterance], tokenizer: PinyinTokenizer,
    n: int = 60, span: tuple[int, int] = (2, 3), min_pos: int = 10,
) -> list[str]:
    """Pick the ``n`` most frequent in-vocab Han n-grams (length in ``span``) with enough
    positives to estimate a per-keyword rate. Frequent words → more positive trials → tighter
    confidence intervals; the ``min_pos`` floor drops keywords too rare to score meaningfully.
    """
    counts: Counter[str] = Counter()
    for u in utts:
        text = "".join(ch for ch in u.text if "一" <= ch <= "鿿")
        for length in range(span[0], span[1] + 1):
            for i in range(len(text) - length + 1):
                counts[text[i : i + length]] += 1
    out: list[str] = []
    for kw, c in counts.most_common():
        if c < min_pos:
            break
        ids = tokenizer.encode(kw)
        if ids and tokenizer.unk_id not in ids:      # fully in-vocab only
            out.append(kw)
        if len(out) >= n:
            break
    return out


def build_trials(
    utts: list[Utterance], keywords: list[str], tokenizer: PinyinTokenizer,
    seed: int = 0, easy_per_pos: int = 1, hard_per_pos: int = 1,
) -> list[Trial]:
    """Build positive, easy-negative and hard-negative trials for every keyword.

    Negatives are sampled to a fixed ratio per positive so the splits stay balanced and no
    single frequent keyword dominates the operating point.
    """
    rng = random.Random(seed)
    by_kw_ids = {kw: tuple(tokenizer.encode(kw)) for kw in keywords}
    trials: list[Trial] = []

    for kw in keywords:
        ids = by_kw_ids[kw]
        pos_utts = [u for u in utts if kw in u.text]
        neg_utts = [u for u in utts if kw not in u.text]
        if not pos_utts or not neg_utts:
            continue

        for u in pos_utts:
            trials.append(Trial(u.utt_id, kw, ids, 1, "pos", u.duration))

        # Easy: this keyword against utterances that do not contain it.
        for u in pos_utts:
            for _ in range(easy_per_pos):
                nu = rng.choice(neg_utts)
                trials.append(Trial(nu.utt_id, kw, ids, 0, "easy", nu.duration))

        # Hard: a confusable of this keyword, tested against utterances that DO contain the
        # real keyword — the audio is close to the query but the query was not said.
        variants = [v for v in tone_confusables(kw, tokenizer) if v[0] != kw]
        if variants:
            for u in pos_utts:
                for _ in range(hard_per_pos):
                    label, vids = variants[rng.randrange(len(variants))]
                    if label in u.text:            # skip if the confusable is actually present
                        continue
                    trials.append(Trial(u.utt_id, label, tuple(vids), 0, "hard", u.duration))
    return trials
