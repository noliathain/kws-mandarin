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


def test_no_nan_on_silence():
    fe = LogMelFrontend()
    feat = fe(torch.zeros(1, 16000))
    assert torch.isfinite(feat).all()


def test_finite_and_no_grad_needed_for_export():
    fe = LogMelFrontend().eval()
    with torch.no_grad():
        feat = fe(torch.randn(1, 16000))
    assert torch.isfinite(feat).all()
