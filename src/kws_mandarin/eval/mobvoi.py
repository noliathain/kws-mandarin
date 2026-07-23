"""MobvoiHotwords (OpenSLR 87) trial construction for zero-shot wake-word evaluation.

MobvoiHotwords is the standard *real-speech* Mandarin KWS benchmark that SOTA papers report
on. It has two hotwords and a large negative set. Our open-vocab model is evaluated ZERO-SHOT:
it never trains on Mobvoi; we simply score every utterance against the hotword's unit sequence
and sweep the threshold. That is the honest test of open-vocab generalization, and it produces
recall@FA / EER / AUC directly comparable to the published Mobvoi numbers.

Metadata schema (from the OpenSLR resources tarball):
  p_{split}.json : positives, each with keyword_id in {0, 1} and utt_id
  n_{split}.json : negatives, keyword_id == -1
keyword_id 0 = "Hi Xiaowen" (你好小问), 1 = "Nihao Wenwen" (你好问问) per the corpus README.
"""

from __future__ import annotations

# keyword_id -> Han text. Verify against the README shipped in the audio tarball before trusting
# a headline number; the eval prints this mapping so a swap would be caught immediately.
HOTWORDS = {0: "你好小问", 1: "你好问问"}


def build_mobvoi_trials(p_entries: list[dict], n_entries: list[dict], keyword_id: int):
    """Return ``[(utt_id, label)]`` for one hotword: label 1 for that hotword's positives, 0 for
    the negative set. The OTHER hotword's positives are excluded, matching the standard per-
    keyword Mobvoi protocol (each hotword is its own detection task against the shared negatives).
    """
    if keyword_id not in HOTWORDS:
        raise ValueError(f"unknown keyword_id {keyword_id}; expected one of {sorted(HOTWORDS)}")
    trials = [(e["utt_id"], 1) for e in p_entries if e["keyword_id"] == keyword_id]
    trials += [(e["utt_id"], 0) for e in n_entries]
    return trials
