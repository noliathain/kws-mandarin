import torch

from kws_mandarin.decode import (
    CTCKeywordSpotter,
    ctc_keyword_score,
    keyword_score_batch,
)
from kws_mandarin.models import KWSModel


def _peaky_posterior():
    # V=6, blank=0, tokens 1..5. Mostly blank, with token 1 peaking at frame 5 and token 2
    # at frame 9 -> the keyword [1, 2] is clearly present, in that order.
    T, V = 15, 6
    logits = torch.zeros(T, V)
    logits[:, 0] = 5.0            # blank dominates elsewhere
    logits[5, 0] = 0.0
    logits[5, 1] = 10.0          # token 1
    logits[9, 0] = 0.0
    logits[9, 2] = 10.0          # token 2
    return logits.log_softmax(-1)


def test_present_keyword_scores_higher_than_absent():
    lp = _peaky_posterior()
    present = ctc_keyword_score(lp, [1, 2])
    absent = ctc_keyword_score(lp, [3, 4])
    assert present > absent


def test_order_matters():
    lp = _peaky_posterior()
    forward = ctc_keyword_score(lp, [1, 2])   # token1 before token2 -> matches
    reversed_ = ctc_keyword_score(lp, [2, 1])  # wrong order
    assert forward > reversed_


def test_partial_match_between_present_and_absent():
    lp = _peaky_posterior()
    full = ctc_keyword_score(lp, [1, 2])
    partial = ctc_keyword_score(lp, [1, 4])   # first unit present, second not
    absent = ctc_keyword_score(lp, [3, 4])
    assert full > partial > absent


def test_spotter_interface_returns_float():
    lp = _peaky_posterior()
    spotter = CTCKeywordSpotter()
    s = spotter.score(lp, [1, 2])
    assert isinstance(s, float) and s == float(ctc_keyword_score(lp, [1, 2]))


def test_batched_score_equals_per_utterance():
    # The vectorized batch DP must match per-utterance scoring of each trimmed sequence.
    torch.manual_seed(0)
    B, T, V = 4, 16, 7
    lp = torch.randn(B, T, V).log_softmax(-1)
    lengths = torch.tensor([16, 9, 13, 5])
    kw = [1, 3, 2]
    batched = keyword_score_batch(lp, lengths, kw)
    for b in range(B):
        single = ctc_keyword_score(lp[b, : int(lengths[b])], kw)
        assert torch.allclose(batched[b], single, atol=1e-4), f"utt {b}: {batched[b]} vs {single}"


def test_end_to_end_model_to_spotter_smoke():
    torch.manual_seed(0)
    model = KWSModel(vocab_size=30, scale=3.0).eval()
    with torch.no_grad():
        lp = model.log_probs(torch.randn(1, 16000))[0]  # (T, V) for one utterance
    score = CTCKeywordSpotter().score(lp, [3, 7, 11])
    assert score == score  # finite (not NaN); untrained model, value itself is meaningless
