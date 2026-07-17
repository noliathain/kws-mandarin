import soundfile as sf
import torch
from torch.utils.data import DataLoader

from kws_mandarin.data import KWSDataset, Utterance, collate_kws, write_manifest
from kws_mandarin.loss import CTCLoss
from kws_mandarin.models import KWSModel
from kws_mandarin.tokenizer import PinyinTokenizer


def _make_corpus(tmp_path):
    specs = [
        ("u1", "你好世界", 1.5),
        ("u2", "打开空调", 1.2),
        ("u3", "播放音乐", 0.9),
    ]
    utts = []
    for uid, text, dur in specs:
        wav = (torch.randn(int(dur * 16000)) * 0.1).numpy()
        path = tmp_path / f"{uid}.wav"
        sf.write(str(path), wav, 16000)
        utts.append(Utterance(uid, str(path), text, dur, "S0", "train"))
    mpath = tmp_path / "train.jsonl"
    write_manifest(mpath, utts)
    return mpath


def test_dataset_and_collate_shapes(tmp_path):
    mpath = _make_corpus(tmp_path)
    tok = PinyinTokenizer()
    ds = KWSDataset(str(mpath), tok)
    assert len(ds) == 3

    loader = DataLoader(ds, batch_size=3, collate_fn=collate_kws)
    batch = next(iter(loader))
    assert batch["wavs"].shape[0] == 3
    assert batch["wavs"].shape[1] == int(1.5 * 16000)      # padded to longest
    assert batch["wav_lengths"].tolist() == [24000, 19200, 14400]
    assert batch["targets"].numel() == int(batch["target_lengths"].sum())


def test_full_training_step(tmp_path):
    mpath = _make_corpus(tmp_path)
    tok = PinyinTokenizer()
    ds = KWSDataset(str(mpath), tok)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_kws)
    model = KWSModel(vocab_size=tok.vocab_size, scale=3.0)
    criterion = CTCLoss(blank=tok.blank_id)

    batch = next(iter(loader))
    logits = model(batch["wavs"])                          # (B, T, V)
    input_lengths = model.output_lengths(batch["wav_lengths"])
    loss = criterion(logits, batch["targets"], input_lengths, batch["target_lengths"])
    loss.backward()

    assert torch.isfinite(loss)
    assert input_lengths.max() <= logits.shape[1]
    # CTC feasibility: every input length must cover its target length
    assert (input_lengths >= batch["target_lengths"]).all()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
