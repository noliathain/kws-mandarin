import wave

from kws_mandarin.data import read_manifest
from kws_mandarin.data.prepare_aishell import build_manifests, load_transcripts


def _write_silence_wav(path, seconds, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)


def _make_fake_aishell(root):
    # transcript: "<utt_id> 甲 醛 白 血 病"
    tx = root / "transcript" / "aishell_transcript_v0.8.txt"
    tx.parent.mkdir(parents=True, exist_ok=True)
    tx.write_text(
        "BAC009S0001W0001 你 好 世 界\n"
        "BAC009S0002W0001 打 开 空 调\n"
        "BAC009S0002W0002 没 有 对 应 音 频\n",  # no wav for this one -> counted missing? no: wav missing
        encoding="utf-8",
    )
    _write_silence_wav(root / "wav" / "train" / "S0001" / "BAC009S0001W0001.wav", 1.5)
    _write_silence_wav(root / "wav" / "train" / "S0002" / "BAC009S0002W0001.wav", 2.0)
    # a wav with no transcript entry -> counted as missing_transcript
    _write_silence_wav(root / "wav" / "dev" / "S0003" / "BAC009S0003W0001.wav", 1.0)


def test_load_transcripts_strips_separators(tmp_path):
    _make_fake_aishell(tmp_path)
    tx = load_transcripts(tmp_path)
    assert tx["BAC009S0001W0001"] == "你好世界"


def test_build_manifests_end_to_end(tmp_path):
    _make_fake_aishell(tmp_path)
    out = tmp_path / "manifests"
    report = build_manifests(tmp_path, out, workers=1)

    train = read_manifest(out / "aishell1_train.jsonl")
    assert len(train) == 2
    by_id = {u.utt_id: u for u in train}
    assert by_id["BAC009S0001W0001"].text == "你好世界"
    assert by_id["BAC009S0001W0001"].duration == 1.5
    assert by_id["BAC009S0001W0001"].speaker == "S0001"
    assert by_id["BAC009S0002W0001"].duration == 2.0

    # dev wav has no transcript -> dropped and counted
    dev = read_manifest(out / "aishell1_dev.jsonl")
    assert len(dev) == 0
    assert report["dev"]["missing_transcript"] == 1


def test_manifest_text_tokenizes_without_unk(tmp_path):
    # The prepared text must round-trip through the tokenizer with no <unk>.
    from kws_mandarin.tokenizer import PinyinTokenizer

    _make_fake_aishell(tmp_path)
    out = tmp_path / "manifests"
    build_manifests(tmp_path, out, workers=1)
    tok = PinyinTokenizer()
    for u in read_manifest(out / "aishell1_train.jsonl"):
        ids = tok.encode(u.text)
        assert ids and tok.unk_id not in ids
