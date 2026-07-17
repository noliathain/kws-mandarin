import torch

from kws_mandarin.features import LogMelFrontend
from kws_mandarin.models import BCResNetEncoder, SubSpectralNorm


def test_subspectralnorm_preserves_shape():
    ssn = SubSpectralNorm(channels=8, sub_bands=5)
    x = torch.randn(2, 8, 20, 30)
    assert ssn(x).shape == x.shape


def test_subspectralnorm_rejects_indivisible_freq():
    ssn = SubSpectralNorm(channels=8, sub_bands=5)
    try:
        ssn(torch.randn(2, 8, 22, 30))  # 22 not divisible by 5
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_encoder_output_shape_and_time_preserved():
    enc = BCResNetEncoder(n_mels=40, scale=3.0).eval()
    T = 101
    feats = torch.randn(2, 40, T)
    out = enc(feats)
    assert out.shape == (2, T, enc.enc_dim)  # time rate preserved (10 ms/frame)


def test_param_budget_is_small():
    enc = BCResNetEncoder(n_mels=40, scale=3.0)
    n = enc.num_params()
    # low-parameter backbone: scale-3 should stay well under 1M params
    assert n < 1_000_000, f"encoder too big: {n} params"


def test_frontend_to_encoder_integration():
    fe = LogMelFrontend()
    enc = BCResNetEncoder(n_mels=fe.output_dim, scale=3.0).eval()
    wav = torch.randn(2, 16000)
    out = enc(fe(wav))
    assert out.shape == (2, fe.num_frames(16000), enc.enc_dim)


def test_backward_runs():
    enc = BCResNetEncoder(n_mels=40, scale=3.0)
    feats = torch.randn(2, 40, 50, requires_grad=True)
    enc(feats).sum().backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()


def test_causal_no_future_leakage():
    # In eval mode, output frame t must depend only on input frames <= t.
    torch.manual_seed(0)
    enc = BCResNetEncoder(n_mels=40, scale=3.0, causal=True).eval()
    T, split = 60, 30
    base = torch.randn(1, 40, T)
    x1 = base.clone()
    x2 = base.clone()
    x2[:, :, split:] = torch.randn(1, 40, T - split)  # differ only in the future
    with torch.no_grad():
        o1, o2 = enc(x1), enc(x2)
    # frames before the split must be identical
    assert torch.allclose(o1[:, :split], o2[:, :split], atol=1e-5)
    # and the change must actually affect the future (sanity: not a constant function)
    assert not torch.allclose(o1[:, split:], o2[:, split:], atol=1e-5)


def test_noncausal_leaks_future_by_design():
    torch.manual_seed(0)
    enc = BCResNetEncoder(n_mels=40, scale=3.0, causal=False).eval()
    T, split = 60, 30
    base = torch.randn(1, 40, T)
    x2 = base.clone()
    x2[:, :, split:] = torch.randn(1, 40, T - split)
    with torch.no_grad():
        o1, o2 = enc(base), enc(x2)
    # non-causal: future edits bleed backward across the split
    assert not torch.allclose(o1[:, :split], o2[:, :split], atol=1e-5)
