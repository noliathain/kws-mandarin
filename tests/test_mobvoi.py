import pytest

from kws_mandarin.eval.mobvoi import HOTWORDS, build_mobvoi_trials
from kws_mandarin.tokenizer import PinyinTokenizer


def test_trials_select_one_hotword_against_negatives():
    # Each hotword is scored against the shared negative set; the OTHER hotword's positives are
    # excluded. Mislabelling here would corrupt recall@FA and every SOTA comparison built on it.
    p = [{"utt_id": "a", "keyword_id": 0}, {"utt_id": "b", "keyword_id": 1},
         {"utt_id": "c", "keyword_id": 0}]
    n = [{"utt_id": "x", "keyword_id": -1}, {"utt_id": "y", "keyword_id": -1}]

    t0 = build_mobvoi_trials(p, n, 0)
    assert ("a", 1) in t0 and ("c", 1) in t0
    assert ("b", 1) not in t0 and ("b", 0) not in t0
    assert ("x", 0) in t0 and ("y", 0) in t0
    assert sum(l for _, l in t0) == 2

    t1 = build_mobvoi_trials(p, n, 1)
    assert ("b", 1) in t1 and sum(l for _, l in t1) == 1


def test_unknown_keyword_id_raises():
    with pytest.raises(ValueError, match="keyword_id"):
        build_mobvoi_trials([], [], 5)


def test_hotwords_are_in_vocab():
    # The model can only score a hotword whose units are all in the tokenizer -- otherwise the
    # zero-shot eval silently scores <unk> and the number is meaningless.
    tok = PinyinTokenizer()
    for text in HOTWORDS.values():
        ids = tok.encode(text)
        assert ids and tok.unk_id not in ids, f"{text} not fully in vocab"
