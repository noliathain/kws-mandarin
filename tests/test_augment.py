import math
import random

import torch

from kws_mandarin.data import SpecAugment, WaveformAugment, add_noise, apply_rir
from kws_mandarin.data.augment import gain_perturb, speed_perturb


def _measure_snr(clean, noise_component):
    return 10.0 * math.log10(clean.pow(2).mean() / noise_component.pow(2).mean())


def test_add_noise_achieves_target_snr():
    rng = random.Random(0)
    torch.manual_seed(0)
    clean = torch.randn(16000)
    noise = torch.randn(16000) * 3.0  # different level -> scaling must correct for it
    for target in (0.0, 5.0, 10.0, 20.0):
        noisy = add_noise(clean, noise, target, rng)
        measured = _measure_snr(clean, noisy - clean)  # added component = noisy - clean
        assert abs(measured - target) < 0.1, f"target {target}, measured {measured}"


def test_add_noise_preserves_length():
    rng = random.Random(0)
    clean = torch.randn(12000)
    short_noise = torch.randn(4000)  # shorter than signal -> must tile
    assert add_noise(clean, short_noise, 10.0, rng).numel() == 12000


def test_apply_rir_preserves_length_and_finite():
    torch.manual_seed(0)
    wav = torch.randn(16000)
    rir = torch.zeros(400)
    rir[10] = 1.0
    rir[50:] = torch.randn(350) * 0.1  # direct path + tail
    out = apply_rir(wav, rir)
    assert out.numel() == 16000
    assert torch.isfinite(out).all()


def test_rir_unit_impulse_is_near_identity():
    # A unit impulse RIR (single tap) should return the signal essentially unchanged.
    wav = torch.randn(8000)
    rir = torch.zeros(64)
    rir[0] = 1.0
    out = apply_rir(wav, rir)
    assert torch.allclose(out, wav, atol=1e-5)


def test_speed_perturb_changes_length_in_right_direction():
    wav = torch.randn(16000)
    faster = speed_perturb(wav, 1.1, 16000)
    slower = speed_perturb(wav, 0.9, 16000)
    assert faster.numel() < 16000 < slower.numel()
    assert speed_perturb(wav, 1.0, 16000).numel() == 16000


def test_gain_perturb_scales_amplitude():
    wav = torch.ones(100)
    assert torch.allclose(gain_perturb(wav, 6.0), wav * 10 ** (6 / 20))


def test_specaugment_masks_only_in_train_mode():
    feat = torch.ones(1, 40, 100)
    sa = SpecAugment(freq_mask=8, n_freq=2, time_mask=25, n_time=2)
    sa.eval()
    assert torch.equal(sa(feat), feat)          # no-op in eval
    sa.train()
    torch.manual_seed(0)
    out = sa(feat)
    assert (out == 0).any() and not torch.equal(out, feat)  # some region masked


def test_waveform_augment_pipeline_runs_and_is_seeded():
    torch.manual_seed(0)
    noise = torch.randn(20000)
    rir = torch.zeros(200)
    rir[5] = 1.0
    aug = WaveformAugment(
        noise_sampler=lambda: noise, rir_sampler=lambda: rir,
        p_noise=1.0, p_rir=1.0, p_speed=1.0, p_gain=1.0, seed=42,
    )
    wav = torch.randn(16000)
    out1 = aug(wav)
    aug2 = WaveformAugment(
        noise_sampler=lambda: noise, rir_sampler=lambda: rir,
        p_noise=1.0, p_rir=1.0, p_speed=1.0, p_gain=1.0, seed=42,
    )
    out2 = aug2(wav)
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2)  # same seed -> reproducible


def test_waveform_augment_can_be_disabled():
    aug = WaveformAugment(p_noise=0.0, p_rir=0.0, p_speed=0.0, p_gain=0.0)
    wav = torch.randn(16000)
    assert torch.equal(aug(wav), wav)  # all probs 0 -> identity
