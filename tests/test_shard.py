import io
import time

import pytest
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


def test_pcm_shard_roundtrip(tmp_path):
    # PCM shards must recover every utterance, decoded to ~the original waveform (int16 precision).
    utts = _corpus(tmp_path, 6)
    shards = write_shards(utts, str(tmp_path / "pcm"), num_shards=2, workers=1, fmt="pcm")
    tok = PinyinTokenizer()
    got = {i["utt_id"]: i for i in ShardDataset(shards, tok, shuffle_buffer=1, shuffle_shards=False)}
    assert set(got) == {u.utt_id for u in utts}
    u0 = next(u for u in utts if u.utt_id == "u0")
    orig, _ = sf.read(u0.wav, dtype="float32")
    assert got["u0"]["wav"].numel() == len(orig)
    assert (got["u0"]["wav"] - torch.from_numpy(orig)).abs().max() < 1e-3
    assert got["u0"]["target"].tolist() == tok.encode(u0.text)


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


def test_bucketing_makes_batches_length_homogeneous(tmp_path):
    # Bucketing exists to cut padding waste: with mixed durations in a batch, every short clip
    # is padded to the longest one and the model burns compute on (and normalizes over) silence.
    # Batches drawn from a length-sorted pool must be far tighter in length than random ones.
    utts = []
    for i in range(64):
        dur = 0.5 + 0.05 * (i % 32)              # 0.5 s .. 2.05 s, interleaved order
        p = tmp_path / f"b{i}.wav"
        sf.write(str(p), (torch.randn(int(dur * 16000)) * 0.1).numpy(), 16000)
        utts.append(Utterance(f"b{i}", str(p), "你好世界", round(dur, 3), f"S{i}", "train"))
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=2, workers=1)
    tok = PinyinTokenizer()

    def spread(ds, bs=8):
        items = list(ds)
        assert len(items) == len(utts)           # bucketing must not drop or duplicate samples
        lens = [it["wav"].numel() for it in items]
        batches = [lens[i:i + bs] for i in range(0, len(lens), bs)]
        # mean padding waste: fraction of the padded batch that is zeros
        return sum(1 - sum(b) / (len(b) * max(b)) for b in batches) / len(batches)

    plain = spread(ShardDataset(shards, tok, shuffle_buffer=32, shuffle_shards=False, seed=0))
    bucketed = spread(ShardDataset(shards, tok, shuffle_buffer=64, shuffle_shards=False, seed=0,
                                   bucket_size=64, batch_size=8))
    assert bucketed < plain / 3                  # padding waste collapses
    assert bucketed < 0.10


def test_parallel_shard_reads_overlap_and_lose_nothing(tmp_path):
    # The storage mount gives ~34 MB/s on one stream but ~285 MB/s across 16, so reading
    # shards sequentially made I/O 90% of the training step. Readers must genuinely overlap,
    # and interleaving must not drop or duplicate a single sample.
    import time

    from kws_mandarin.data.shard import _interleave_shards, _iter_shard_raw

    utts = _corpus(tmp_path, 24)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=8, workers=1)

    ids = sorted(item[0] for item in _interleave_shards(shards, 4))
    assert ids == sorted(u.utt_id for u in utts)      # complete, no duplicates

    # overlap: simulate a slow mount by timing a delayed read serially vs interleaved
    delay = 0.05
    def slow(path):
        time.sleep(delay)
        yield from _iter_shard_raw(path)

    t0 = time.perf_counter()
    for p in shards:
        list(slow(p))
    serial = time.perf_counter() - t0

    import kws_mandarin.data.shard as sh
    real, sh._iter_shard_raw = sh._iter_shard_raw, slow
    try:
        t0 = time.perf_counter()
        list(_interleave_shards(shards, 8))
        parallel = time.perf_counter() - t0
    finally:
        sh._iter_shard_raw = real

    assert parallel < serial / 2, f"reads did not overlap: {parallel:.3f}s vs {serial:.3f}s"


def test_interleave_shards_stops_readers_when_consumer_quits(tmp_path):
    # Training stops mid-stream (max_steps, bucketing). Reader threads must not stay blocked
    # on a full queue -- leaked threads would pin memory and file handles for the whole run.
    import threading

    from kws_mandarin.data.shard import _interleave_shards

    utts = _corpus(tmp_path, 40)
    shards = write_shards(utts, str(tmp_path / "shards"), num_shards=4, workers=1)

    before = threading.active_count()
    gen = _interleave_shards(shards, 4)
    next(gen)                                          # start readers, then abandon them
    gen.close()
    for _ in range(100):
        if threading.active_count() <= before:
            break
        time.sleep(0.05)
    assert threading.active_count() <= before, "reader threads leaked after consumer stopped"
