import soundfile as sf
import torch

from kws_mandarin.data import WaveformAugment, load_rir_pack, pack_rirs


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
