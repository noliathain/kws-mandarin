from kws_mandarin.data.manifest import Utterance
from kws_mandarin.eval.benchmark import build_trials, select_keywords
from kws_mandarin.tokenizer import PinyinTokenizer


def _utts(specs):
    return [Utterance(f"u{i}", f"/tmp/u{i}.wav", t, 2.0, f"S{i}", "test")
            for i, t in enumerate(specs)]


def test_select_keywords_picks_frequent_in_vocab_ngrams():
    tok = PinyinTokenizer()
    utts = _utts(["你好世界"] * 12 + ["打开空调"] * 12 + ["罕见词"] * 2)
    kws = select_keywords(utts, tok, n=10, span=(2, 2), min_pos=10)
    assert "你好" in kws and "打开" in kws
    # "罕见" appears only twice (< min_pos) -> excluded
    assert "罕见" not in kws


def test_trials_are_correctly_labelled():
    # A positive trial's keyword really is in the utterance; a negative's is not. If labels are
    # wrong, EER is meaningless.
    tok = PinyinTokenizer()
    utts = _utts(["你好世界"] * 10 + ["打开空调"] * 10)
    trials = build_trials(utts, ["你好"], tok, seed=0)
    for t in trials:
        contains = any(t.query in u.text for u in utts if u.utt_id == t.utt_id)
        assert (t.label == 1) == contains, f"{t.split} trial mislabelled"


def test_hard_negatives_are_confusable_and_absent():
    # Hard negatives must be a confusable of the real keyword AND not actually present -- that
    # is the whole point of the Hard split. A hard neg equal to the true keyword would be a
    # false negative label.
    tok = PinyinTokenizer()
    utts = _utts(["中国经济"] * 20 + ["天气很好"] * 10)   # some utts lack 中国 (real corpus)
    trials = build_trials(utts, ["中国"], tok, seed=1, hard_per_pos=2)
    hard = [t for t in trials if t.split == "hard"]
    assert hard, "no hard negatives generated"
    for t in hard:
        assert t.query != "中国"                       # it's a variant, not the keyword
        assert t.label == 0
        assert t.query not in "中国经济"                 # genuinely absent from the audio's text


def test_trials_deterministic_given_seed():
    tok = PinyinTokenizer()
    utts = _utts(["你好世界", "打开空调", "你好朋友"] * 8)
    a = build_trials(utts, ["你好"], tok, seed=7)
    b = build_trials(utts, ["你好"], tok, seed=7)
    assert [(t.utt_id, t.query, t.label, t.split) for t in a] == \
           [(t.utt_id, t.query, t.label, t.split) for t in b]
