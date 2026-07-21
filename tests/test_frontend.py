import torch

from kws_mandarin.features import LogMelFrontend


def test_output_shape_1s():
    fe = LogMelFrontend()
    wav = torch.randn(2, 16000)  # 2 x 1 second
    feat = fe(wav)
    assert feat.shape == (2, 40, fe.num_frames(16000))
    assert feat.shape[2] == 101


def test_accepts_channel_dim():
    fe = LogMelFrontend()
    feat = fe(torch.randn(3, 1, 8000))
    assert feat.shape == (3, 40, fe.num_frames(8000))


def test_per_utt_norm_zero_mean_unit_std():
    fe = LogMelFrontend(per_utt_norm=True)
    feat = fe(torch.randn(1, 16000))
    # after CMVN over time, each mel channel is ~zero-mean / ~unit-std
    assert feat.mean(dim=-1).abs().max() < 1e-4
    assert (feat.std(dim=-1) - 1.0).abs().max() < 1e-2


def test_masked_cmvn_ignores_padding():
    # A clip's features must not change when it's padded in a longer batch. Without masking,
    # CMVN statistics are dominated by the padding silence and distort the real speech.
    torch.manual_seed(0)
    fe = LogMelFrontend(per_utt_norm=True)
    wav = torch.randn(16000) * 0.1
    solo = fe(wav.unsqueeze(0), torch.tensor([16000]))

    padded = torch.zeros(1, 48000)
    padded[0, :16000] = wav
    masked = fe(padded, torch.tensor([16000]))

    n = solo.shape[-1] - 2                      # skip the STFT frames that straddle the edge
    unmasked = fe(padded)                       # what we did before the fix
    err_masked = (solo[:, :, :n] - masked[:, :, :n]).abs().mean()
    err_unmasked = (solo[:, :, :n] - unmasked[:, :, :n]).abs().mean()

    assert err_masked < err_unmasked / 20       # padding no longer drives the CMVN statistics
    assert err_masked < 0.05                    # residual is the boundary frame's half-zero window
    assert masked[:, :, solo.shape[-1]:].abs().max() < 1e-6      # padding zeroed


def test_no_nan_on_silence():
    fe = LogMelFrontend()
    feat = fe(torch.zeros(1, 16000))
    assert torch.isfinite(feat).all()


def test_finite_and_no_grad_needed_for_export():
    fe = LogMelFrontend().eval()
    with torch.no_grad():
        feat = fe(torch.randn(1, 16000))
    assert torch.isfinite(feat).all()
