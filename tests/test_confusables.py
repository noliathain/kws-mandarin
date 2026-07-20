import torch

from kws_mandarin.confusables import confusion_margin, tone_confusables
from kws_mandarin.decode import CTCKeywordSpotter
from kws_mandarin.tokenizer import PinyinTokenizer


def test_generates_tone_swap_confusables():
    tok = PinyinTokenizer()
    confs = tone_confusables("打开", tok)  # da3 kai1
    labels = [p for p, _ in confs]
    # da3 has tone-3 -> tone-2 confusion must appear
    assert any("da2" in lb for lb in labels), labels
    # neutral-tone confusion of kai1 -> kai5
    assert any("kai5" in lb for lb in labels), labels


def test_confusables_are_in_vocab_and_distinct():
    tok = PinyinTokenizer()
    original = tuple(tok.encode("经济"))
    for _, ids in tone_confusables("经济", tok):
        assert tok.unk_id not in ids          # in-vocab
        assert tuple(ids) != original          # never the target itself
        assert len(ids) > 0


def test_initial_confusion_zh_z():
    tok = PinyinTokenizer()
    # 中(zhong1): zh->z confusion should be generated
    labels = [p for p, _ in tone_confusables("中国", tok)]
    assert any(lb.startswith("zong1") or " zong1" in lb or lb.split()[0] == "zong1"
               for lb in labels), labels


def test_max_variants_capped():
    tok = PinyinTokenizer()
    assert len(tone_confusables("经济发展", tok, max_variants=5)) <= 5


def test_empty_for_non_han():
    tok = PinyinTokenizer()
    assert tone_confusables("hello123", tok) == []


def test_confusion_margin_positive_when_target_present():
    # Build a posterior that cleanly spells the target's units; the target must then out-score
    # every tone-confusable (positive margin).
    tok = PinyinTokenizer()
    text = "打开"
    unit_ids = tok.encode(text)
    V = tok.vocab_size
    T = 2 * len(unit_ids) + 4
    logits = torch.zeros(T, V)
    logits[:, tok.blank_id] = 5.0
    for k, uid in enumerate(unit_ids):
        f = 2 * k + 1
        logits[f, tok.blank_id] = 0.0
        logits[f, uid] = 12.0
    lp = logits.log_softmax(-1)

    margin = confusion_margin(lp, text, tok, CTCKeywordSpotter(blank=tok.blank_id))
    assert margin is not None and margin > 0
