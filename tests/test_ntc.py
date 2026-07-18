import torch

from kws_mandarin.decode import (
    NTCKeywordSpotter,
    ctc_keyword_score,
    ntc_keyword_score,
)


def post(T, V, spec):
    """Build (T, V) log-probs. Frames not in spec are blank-dominant; a spec entry
    {frame: {token: logit}} fully overrides that frame (blank logit defaults to 0)."""
    logits = torch.zeros(T, V)
    logits[:, 0] = 5.0
    for t, d in spec.items():
        logits[t] = torch.zeros(V)
        for tok, val in d.items():
            logits[t, tok] = float(val)
    return logits.log_softmax(-1)


def test_ntc_reduces_to_ctc_when_wildcard_disabled():
    # Correctness anchor: as lambda -> large, the wildcard vanishes and NTC == plain CTC.
    torch.manual_seed(0)
    lp = torch.randn(20, 8).log_softmax(-1)
    for kw in ([1, 2], [3, 4, 5]):
        ntc = ntc_keyword_score(lp, kw, lambda_ins=50.0, lambda_mask=50.0)
        ctc = ctc_keyword_score(lp, kw)
        assert torch.allclose(ntc, ctc, atol=1e-3), f"{kw}: {ntc.item()} vs {ctc.item()}"


def test_ntc_still_discriminates_on_clean():
    lp = post(15, 6, {4: {1: 10}, 8: {2: 10}})  # keyword [1,2] present in order
    assert ntc_keyword_score(lp, [1, 2]) > ntc_keyword_score(lp, [3, 4])   # present > absent
    assert ntc_keyword_score(lp, [1, 2]) > ntc_keyword_score(lp, [2, 1])   # order matters


def test_ntc_robust_to_noise_insertion():
    # Spurious non-keyword tokens (noise) inserted between the two keyword units.
    clean = post(15, 6, {4: {1: 10}, 8: {2: 10}})
    noisy = post(15, 6, {4: {1: 10}, 8: {2: 10}, 5: {4: 8}, 6: {4: 8}, 7: {4: 8}})

    ctc_clean, ctc_noisy = ctc_keyword_score(clean, [1, 2]), ctc_keyword_score(noisy, [1, 2])
    ntc_clean, ntc_noisy = ntc_keyword_score(clean, [1, 2]), ntc_keyword_score(noisy, [1, 2])

    # NTC's self-loop wildcard absorbs the inserted noise -> higher score under noise...
    assert ntc_noisy > ctc_noisy
    # ...and degrades less than plain CTC from clean to noisy.
    assert (ntc_clean - ntc_noisy) < (ctc_clean - ctc_noisy)


def test_ntc_robust_to_token_masking():
    # The middle keyword unit is masked by noise (a spurious token replaces it).
    clean = post(15, 6, {3: {1: 10}, 7: {2: 10}, 11: {3: 10}})
    masked = post(15, 6, {3: {1: 10}, 11: {3: 10}, 7: {5: 8}})  # unit 2 masked by token 5

    ctc_masked = ctc_keyword_score(masked, [1, 2, 3])
    ntc_masked = ntc_keyword_score(masked, [1, 2, 3])
    # NTC's bypass arc traverses the masked unit -> higher score than plain CTC.
    assert ntc_masked > ctc_masked
    # and the keyword is still detectable above an absent one under masking
    assert ntc_masked > ntc_keyword_score(masked, [4, 4, 4])


def test_larger_penalty_is_less_tolerant():
    noisy = post(15, 6, {4: {1: 10}, 8: {2: 10}, 5: {4: 8}, 6: {4: 8}, 7: {4: 8}})
    lenient = ntc_keyword_score(noisy, [1, 2], lambda_ins=1.0, lambda_mask=1.0)
    strict = ntc_keyword_score(noisy, [1, 2], lambda_ins=6.0, lambda_mask=6.0)
    assert lenient > strict  # smaller penalty absorbs more noise -> higher score


def test_ntc_spotter_interface():
    lp = post(15, 6, {4: {1: 10}, 8: {2: 10}})
    sp = NTCKeywordSpotter()
    s = sp.score(lp, [1, 2])
    assert isinstance(s, float) and s == float(ntc_keyword_score(lp, [1, 2]))
