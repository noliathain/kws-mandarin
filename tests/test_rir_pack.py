import soundfile as sf
import torch

from kws_mandarin.data import (
    WaveformAugment,
    load_noise_pack,
    load_rir_pack,
    pack_noise,
    pack_rirs,
)


def _make_rirs(tmp_path, n=6):
    d = tmp_path / "rirs"
    d.mkdir()
    for i in range(n):
        rir = torch.zeros(200)
        rir[0] = 1.0                      # direct path
        rir[20:] = torch.randn(180) * 0.05  # tail
        sf.write(str(d / f"rir{i}.wav"), rir.numpy(), 16000)
    return str(d)


def test_pack_and_load_roundtrip(tmp_path):
    rir_dir = _make_rirs(tmp_path, 6)
    out = str(tmp_path / "rirs.pt")
    n = pack_rirs(rir_dir, out, max_rirs=100, sample_rate=16000)
    assert n == 6
    rirs = load_rir_pack(out, sample_rate=16000)
    assert len(rirs) == 6 and all(r.numel() == 200 for r in rirs)


def test_max_rirs_caps_count(tmp_path):
    rir_dir = _make_rirs(tmp_path, 10)
    out = str(tmp_path / "rirs.pt")
    assert pack_rirs(rir_dir, out, max_rirs=4) == 4


def test_load_rejects_sample_rate_mismatch(tmp_path):
    rir_dir = _make_rirs(tmp_path, 3)
    out = str(tmp_path / "rirs.pt")
    pack_rirs(rir_dir, out, sample_rate=16000)
    try:
        load_rir_pack(out, sample_rate=8000)
        raise AssertionError("expected ValueError on sr mismatch")
    except ValueError:
        pass


def test_pack_skips_overlong_non_rir_files(tmp_path):
    # OpenSLR 28 mixes real RIRs with 30 s isotropic-noise recordings in one folder;
    # the packer must drop the long ones (they are not impulse responses).
    d = tmp_path / "mixed"
    d.mkdir()
    sf.write(str(d / "rir_short.wav"), (torch.randn(8000) * 0.1).numpy(), 16000)   # 0.5 s RIR
    sf.write(str(d / "noise_long.wav"), (torch.randn(30 * 16000) * 0.1).numpy(), 16000)  # 30 s noise
    out = str(tmp_path / "rirs.pt")
    n = pack_rirs(d.as_posix(), out, max_rir_seconds=2.0)
    assert n == 1                                   # only the short RIR kept
    assert load_rir_pack(out)[0].numel() == 8000


def test_apply_rir_batch_matches_per_clip():
    # Batched GPU-style RIR must equal per-clip apply_rir for each item (correctness anchor).
    from kws_mandarin.data import apply_rir, apply_rir_batch

    torch.manual_seed(0)
    B, T, L = 3, 8000, 200
    wavs = torch.randn(B, T)
    rirs = torch.zeros(B, L)
    for b in range(B):
        rirs[b, 3 + b] = 1.0                       # direct-path peak at a different tap per clip
        rirs[b, 20:] = torch.randn(L - 20) * 0.05
    batched = apply_rir_batch(wavs, rirs)
    assert batched.shape == (B, T)
    for b in range(B):
        single = apply_rir(wavs[b], rirs[b])
        assert torch.allclose(batched[b], single, atol=1e-4), f"clip {b} mismatch"


def test_apply_rir_handles_long_rir_without_oom(tmp_path):
    # Regression: a multi-second RIR must not blow up memory (fftconvolve, not conv1d).
    from kws_mandarin.data.augment import apply_rir

    wav = torch.randn(16000)
    long_rir = torch.randn(2 * 16000) * 0.01  # 2 s
    out = apply_rir(wav, long_rir)
    assert out.numel() == 16000 and torch.isfinite(out).all()


def test_pack_noise_crops_long_and_roundtrips(tmp_path):
    d = tmp_path / "noise"
    d.mkdir()
    sf.write(str(d / "short.wav"), (torch.randn(8000) * 0.1).numpy(), 16000)        # 0.5 s
    sf.write(str(d / "long.wav"), (torch.randn(40 * 16000) * 0.1).numpy(), 16000)   # 40 s
    out = str(tmp_path / "noise.pt")
    n = pack_noise(d.as_posix(), out, max_seconds=15.0)
    assert n == 1                       # 0.5 s clip dropped: tiled to utterance length it buzzes
    assert [x.numel() for x in load_noise_pack(out)] == [15 * 16000]   # long cropped to 15 s

    n = pack_noise(d.as_posix(), out, max_seconds=15.0, min_seconds=0.0)
    assert sorted(x.numel() for x in load_noise_pack(out)) == [8000, 15 * 16000]


def test_waveform_augment_uses_noise_pack_in_memory(tmp_path):
    d = tmp_path / "noise"
    d.mkdir()
    sf.write(str(d / "n0.wav"), (torch.randn(16000) * 0.5).numpy(), 16000)
    out = str(tmp_path / "noise.pt")
    pack_noise(d.as_posix(), out)
    aug = WaveformAugment.from_dirs(
        musan_dir=None, rir_dir=None, noise_pack=out,
        p_noise=1.0, p_rir=0.0, p_speed=0.0, p_gain=0.0, seed=1,
    )
    out_wav = aug(torch.randn(16000))
    assert out_wav.numel() == 16000 and torch.isfinite(out_wav).all()
    assert not torch.equal(out_wav, torch.randn(16000))  # noise was mixed in


def test_waveform_augment_uses_packed_rirs_in_memory(tmp_path):
    rir_dir = _make_rirs(tmp_path, 5)
    out = str(tmp_path / "rirs.pt")
    pack_rirs(rir_dir, out, sample_rate=16000)
    aug = WaveformAugment.from_dirs(
        musan_dir=None, rir_dir=None, rir_pack=out,
        p_noise=0.0, p_speed=0.0, p_gain=0.0, p_rir=1.0, seed=1,
    )
    wav = torch.randn(16000)
    out_wav = aug(wav)
    assert out_wav.numel() == 16000 and torch.isfinite(out_wav).all()
    assert not torch.equal(out_wav, wav)  # reverb was actually applied


def test_pack_noise_crops_on_disk_not_just_in_view(tmp_path):
    # torch.save serializes a tensor's whole storage, so saving slices would write every clip's
    # full-length audio -- a 15 s crop of 3-minute MUSAN files ballooned the pack ~30x.
    import soundfile as sf

    from kws_mandarin.data.rir_pack import load_noise_pack, pack_noise

    src = tmp_path / "noise"
    src.mkdir()
    for i in range(4):
        sf.write(str(src / f"n{i}.wav"), (torch.randn(16000 * 30) * 0.1).numpy(), 16000)

    out = tmp_path / "noise.pt"
    assert pack_noise(str(src), str(out), max_items=4, max_seconds=2.0) == 4
    assert all(n.numel() == 32000 for n in load_noise_pack(str(out)))
    assert out.stat().st_size < 4 * 32000 * 4 * 2      # ~cropped size, not 30 s per clip
