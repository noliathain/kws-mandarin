import soundfile as sf
import torch
from torch.utils.data import DataLoader

from kws_mandarin.data import (
    ShardDataset,
    Utterance,
    collate_kws,
    select_shards,
    write_shards,
)
from kws_mandarin.loss import CTCLoss
from kws_mandarin.models import KWSModel
from kws_mandarin.tokenizer import PinyinTokenizer


def _corpus(tmp_path, n=12):
    texts = ["你好世界", "打开空调", "播放音乐", "今天天气"]
    utts = []
    for i in range(n):
        dur = 0.8 + 0.05 * i
        p = tmp_path / f"u{i}.wav"
        sf.write(str(p), (torch.randn(int(dur * 16000)) * 0.1).numpy(), 16000)
        utts.append(Utterance(f"u{i}", str(p), texts[i % len(texts)], round(dur, 3), f"S{i}", "train"))
    return utts


def test_shard_roundtrip_recovers_all_utterances(tmp_path):
    utts = _corpus(tmp_path, 12)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=4, workers=1)
    assert len(shards) == 4

    tok = PinyinTokenizer()
    ds = ShardDataset(shards, tok, shuffle_buffer=1, shuffle_shards=False)
    seen = {}
    for item in ds:
        seen[item["utt_id"]] = item
    assert set(seen) == {u.utt_id for u in utts}          # nothing lost or duplicated
    # decoded target matches tokenizing the original text
    u0 = next(u for u in utts if u.utt_id == "u0")
    assert seen["u0"]["target"].tolist() == tok.encode(u0.text)


def test_select_shards_partition_is_disjoint_and_complete():
    shards = [f"s{i}" for i in range(10)]
    for world, n_workers in [(1, 1), (2, 3), (4, 2), (8, 8)]:
        covered = []
        for rank in range(world):
            for worker in range(n_workers):
                covered.extend(select_shards(shards, rank, world, worker, n_workers))
        assert sorted(covered) == sorted(shards)          # complete coverage
        assert len(covered) == len(set(covered))          # no shard read twice


def test_shuffle_buffer_preserves_multiset(tmp_path):
    utts = _corpus(tmp_path, 10)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=3, workers=1)
    tok = PinyinTokenizer()
    ds = ShardDataset(shards, tok, shuffle_buffer=4, shuffle_shards=True, seed=1)
    ids = sorted(item["utt_id"] for item in ds)
    assert ids == sorted(u.utt_id for u in utts)          # shuffling loses nothing


def test_infinite_mode_cycles_past_one_pass(tmp_path):
    # In DDP we cycle shards forever so ranks never hit an epoch boundary (which desyncs the
    # all-reduce when shards split unevenly). A finite dataset yields exactly the corpus once;
    # an infinite one keeps going.
    utts = _corpus(tmp_path, 6)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=2, workers=1)
    tok = PinyinTokenizer()

    finite = ShardDataset(shards, tok, shuffle_buffer=1, shuffle_shards=False, infinite=False)
    assert sum(1 for _ in finite) == 6

    inf = ShardDataset(shards, tok, shuffle_buffer=1, shuffle_shards=True, infinite=True)
    it = iter(inf)
    got = [next(it)["utt_id"] for _ in range(15)]   # more than the 6 real utts -> must cycle
    assert len(got) == 15
    assert set(got) == {u.utt_id for u in utts}      # only real utts, just repeated


def test_threaded_loading_recovers_all_samples(tmp_path):
    # Parallel decode+augment across threads must yield the same set of utterances as serial.
    utts = _corpus(tmp_path, 12)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=4, workers=1)
    tok = PinyinTokenizer()

    serial = {i["utt_id"] for i in ShardDataset(shards, tok, shuffle_buffer=1, shuffle_shards=False)}
    threaded = ShardDataset(shards, tok, shuffle_buffer=1, shuffle_shards=False, num_threads=4)
    got = {}
    for item in threaded:
        got[item["utt_id"]] = item
    assert set(got) == serial == {u.utt_id for u in utts}
    # decoded target still correct through the thread path
    u0 = next(u for u in utts if u.utt_id == "u0")
    assert got["u0"]["target"].tolist() == tok.encode(u0.text)


def test_training_step_from_shards(tmp_path):
    utts = _corpus(tmp_path, 8)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=4, workers=1)
    tok = PinyinTokenizer()
    ds = ShardDataset(shards, tok, shuffle_buffer=1)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_kws)
    model = KWSModel(vocab_size=tok.vocab_size, scale=1.5)
    criterion = CTCLoss(blank=tok.blank_id)

    batch = next(iter(loader))
    logits = model(batch["wavs"])
    input_lengths = model.output_lengths(batch["wav_lengths"])
    loss = criterion(logits, batch["targets"], input_lengths, batch["target_lengths"])
    loss.backward()
    assert torch.isfinite(loss)
    assert (input_lengths >= batch["target_lengths"]).all()


def test_augment_hook_is_applied(tmp_path):
    utts = _corpus(tmp_path, 4)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=2, workers=1)
    tok = PinyinTokenizer()
    marker = torch.tensor([42.0])
    ds = ShardDataset(shards, tok, shuffle_buffer=1, augment=lambda w: marker)
    assert all(torch.equal(item["wav"], marker) for item in ds)
