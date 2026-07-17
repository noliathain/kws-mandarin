import soundfile as sf
import torch

from kws_mandarin.data import Utterance, validate_manifests, write_manifest


def _wav(tmp_path, uid, dur):
    path = tmp_path / f"{uid}.wav"
    sf.write(str(path), (torch.randn(max(1, int(dur * 16000))) * 0.1).numpy(), 16000)
    return str(path)


def _manifests(tmp_path, utts_by_split):
    paths = {}
    for split, utts in utts_by_split.items():
        p = tmp_path / f"{split}.jsonl"
        write_manifest(p, utts)
        paths[split] = str(p)
    return paths


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_clean_corpus_passes(tmp_path):
    utts = {
        "train": [Utterance("t1", _wav(tmp_path, "t1", 2.0), "你好世界", 2.0, "S1", "train"),
                  Utterance("t2", _wav(tmp_path, "t2", 1.5), "打开空调", 1.5, "S2", "train")],
        "test": [Utterance("e1", _wav(tmp_path, "e1", 1.8), "播放音乐", 1.8, "S9", "test")],
    }
    report = validate_manifests(_manifests(tmp_path, utts), workers=1, deep=True)
    assert report.ok, report.render()


def test_ctc_infeasibility_is_caught(tmp_path):
    # 0.05 s clip (~5 frames) cannot hold an 8-character target's units -> FAIL.
    utts = {"train": [
        Utterance("bad", _wav(tmp_path, "bad", 0.05), "今天天气怎么样啊", 0.05, "S1", "train"),
        Utterance("ok", _wav(tmp_path, "ok", 2.0), "你好", 2.0, "S2", "train"),
    ]}
    report = validate_manifests(_manifests(tmp_path, utts), workers=1)
    assert not report.ok
    assert _check(report, "ctc_feasibility").status == "FAIL"


def test_speaker_leakage_is_caught(tmp_path):
    # S1 appears in both train and test -> evaluation-invalidating leak.
    utts = {
        "train": [Utterance("t1", _wav(tmp_path, "t1", 2.0), "你好世界", 2.0, "S1", "train")],
        "test": [Utterance("e1", _wav(tmp_path, "e1", 2.0), "打开空调", 2.0, "S1", "test")],
    }
    report = validate_manifests(_manifests(tmp_path, utts), workers=1)
    assert not report.ok
    assert _check(report, "speaker_leakage").status == "FAIL"


def test_missing_wav_is_caught(tmp_path):
    utts = {"train": [Utterance("gone", str(tmp_path / "nope.wav"), "你好", 2.0, "S1", "train")]}
    report = validate_manifests(_manifests(tmp_path, utts), workers=1)
    assert _check(report, "missing_wav").status == "FAIL"


def test_duplicate_utt_id_is_caught(tmp_path):
    utts = {"train": [
        Utterance("dup", _wav(tmp_path, "dup", 2.0), "你好", 2.0, "S1", "train"),
        Utterance("dup", _wav(tmp_path, "dup", 2.0), "世界", 2.0, "S2", "train"),
    ]}
    report = validate_manifests(_manifests(tmp_path, utts), workers=1)
    assert _check(report, "duplicate_utt_id").status == "FAIL"


def test_deep_flags_wrong_sample_rate(tmp_path):
    path = tmp_path / "sr8k.wav"
    sf.write(str(path), (torch.randn(16000) * 0.1).numpy(), 8000)  # 8 kHz, wrong
    utts = {"train": [Utterance("s", str(path), "你好", 2.0, "S1", "train")]}
    report = validate_manifests(_manifests(tmp_path, utts), workers=1, deep=True)
    assert _check(report, "audio_format").status == "FAIL"


def test_summary_has_per_split_stats(tmp_path):
    utts = {"train": [Utterance("t1", _wav(tmp_path, "t1", 2.0), "你好世界", 2.0, "S1", "train")]}
    report = validate_manifests(_manifests(tmp_path, utts), workers=1)
    assert report.summary["splits"]["train"]["utts"] == 1
    assert report.summary["splits"]["train"]["speakers"] == 1
