"""WebDataset-style tar shards for FUSE/S3-friendly training I/O.

Reading 141k individual wavs off an S3-backed FUSE mount every batch is I/O-bound and
fragile. Instead we pack the corpus once into a few hundred large ``.tar`` shards, each
holding many samples as ``<key>.flac`` + ``<key>.json`` member pairs. Training then does
large *sequential* reads, which cloud object storage handles well.

``ShardDataset`` is an ``IterableDataset`` that partitions shards across DDP ranks *and*
DataLoader workers (each shard consumed by exactly one reader — no duplication), with an
in-memory shuffle buffer for sample-level randomness. Yields the same dicts as ``KWSDataset``
so ``collate_kws`` is unchanged.
"""

from __future__ import annotations

import io
import json
import queue
import random
import tarfile
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from ..tokenizer import PinyinTokenizer
from .manifest import Utterance, read_manifest


def _encode_flac(wav: Tensor, sample_rate: int) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, wav.numpy(), sample_rate, format="FLAC", subtype="PCM_16")
    return bio.getvalue()


def _encode_pcm(wav: Tensor, sample_rate: int) -> bytes:
    """Raw little-endian int16 mono PCM. Decoding is a memcpy (no codec) — the fast path."""
    return (wav.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()


def _decode_flac(raw: bytes, sample_rate: int) -> Tensor:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def _decode_pcm(raw: bytes) -> Tensor:
    return torch.from_numpy(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)


def _decode_audio(raw: bytes, ext: str, sample_rate: int) -> Tensor:
    return _decode_pcm(raw) if ext == "pcm" else _decode_flac(raw, sample_rate)


def _load_wav(path: str, sample_rate: int) -> Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


# --- writing --------------------------------------------------------------------------

def _write_one_shard(args: tuple) -> tuple[str, int]:
    shard_path, utt_dicts, sample_rate, fmt = args
    audio_ext = ".pcm" if fmt == "pcm" else ".flac"
    encode = _encode_pcm if fmt == "pcm" else _encode_flac
    n = 0
    with tarfile.open(shard_path, "w") as tar:
        for u in utt_dicts:
            try:
                wav = _load_wav(u["wav"], sample_rate)
            except Exception:
                continue
            key = u["utt_id"]
            audio = encode(wav, sample_rate)
            meta = json.dumps(
                {"text": u["text"], "duration": u["duration"], "speaker": u["speaker"]},
                ensure_ascii=False,
            ).encode("utf-8")
            for ext, payload in ((audio_ext, audio), (".json", meta)):
                info = tarfile.TarInfo(f"{key}{ext}")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            n += 1
    return shard_path, n


def write_shards(
    utterances: list[Utterance],
    out_dir: str,
    prefix: str = "shard",
    num_shards: int = 256,
    sample_rate: int = 16000,
    workers: int = 8,
    fmt: str = "flac",
) -> list[str]:
    """Pack utterances into ``num_shards`` tar shards (round-robin for balance + shuffling).

    ``fmt``: "flac" (compact, ~2x smaller) or "pcm" (raw int16 — no decode at read time, so
    the training loop stays fast without any DataLoader workers).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, min(num_shards, len(utterances)))
    # round-robin assignment balances shard sizes and pre-shuffles speakers across shards
    buckets: list[list[dict]] = [[] for _ in range(num_shards)]
    for i, u in enumerate(utterances):
        buckets[i % num_shards].append(asdict(u))

    tasks = [
        (str(out / f"{prefix}-{s:05d}.tar"), buckets[s], sample_rate, fmt)
        for s in range(num_shards)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_write_one_shard, tasks))
    else:
        results = [_write_one_shard(t) for t in tasks]
    return [p for p, _ in sorted(results)]


# --- reading --------------------------------------------------------------------------

def select_shards(shards: list[str], rank: int, world: int, worker: int, n_workers: int) -> list[str]:
    """Deterministic disjoint partition of shards across (rank, worker) — no duplication."""
    global_id = rank * n_workers + worker
    total = world * n_workers
    return shards[global_id::total]


def _iter_shard(path: str, sample_rate: int):
    """Yield (utt_id, wav, text, duration) grouped by key from one tar shard."""
    with tarfile.open(path, "r") as tar:
        cur_key = None
        cur: dict[str, bytes] = {}
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            key, ext = name.rsplit(".", 1)
            data = tar.extractfile(member).read()
            if cur_key is not None and key != cur_key:
                yield _assemble(cur_key, cur, sample_rate)
                cur = {}
            cur_key = key
            cur[ext] = data
        if cur_key is not None:
            yield _assemble(cur_key, cur, sample_rate)


def _audio_part(cur: dict[str, bytes]) -> tuple[str, bytes]:
    ext = "pcm" if "pcm" in cur else "flac"
    return ext, cur[ext]


def _assemble(key: str, parts: dict[str, bytes], sample_rate: int):
    meta = json.loads(parts["json"].decode("utf-8"))
    ext, audio = _audio_part(parts)
    return key, _decode_audio(audio, ext, sample_rate), meta["text"], meta["duration"]


def _raw_item(key: str, cur: dict[str, bytes]):
    ext, audio = _audio_part(cur)
    return key, ext, audio, json.loads(cur["json"])["text"]


def _iter_shard_raw(path: str):
    """Yield (key, audio_ext, audio_bytes, text) grouped by key — cheap sequential tar reads,
    NO decode. Decode/augment happens downstream (per-sample), where it can be batched/parallel.
    """
    with tarfile.open(path, "r") as tar:
        cur_key = None
        cur: dict[str, bytes] = {}
        for member in tar:
            if not member.isfile():
                continue
            key, ext = member.name.rsplit(".", 1)
            data = tar.extractfile(member).read()
            if cur_key is not None and key != cur_key:
                yield _raw_item(cur_key, cur)
                cur = {}
            cur_key = key
            cur[ext] = data
        if cur_key is not None:
            yield _raw_item(cur_key, cur)


def _threaded_map(fn, iterable, num_threads: int):
    """Stream ``fn`` over ``iterable`` across a thread pool, order-preserving, bounded in-flight.

    FLAC decode + torchaudio resample release the GIL, so this genuinely parallelizes the
    per-clip work across cores — with no multiprocessing (avoids the container's broken
    worker->main shared-memory IPC). Results that come back None are dropped.
    """
    max_pending = num_threads * 3
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        it = iter(iterable)
        pending: deque = deque()
        for _ in range(max_pending):
            try:
                pending.append(ex.submit(fn, next(it)))
            except StopIteration:
                break
        while pending:
            res = pending.popleft().result()
            try:
                pending.append(ex.submit(fn, next(it)))
            except StopIteration:
                pass
            if res is not None:
                yield res


def _interleave_shards(paths: list[str], num_streams: int):
    """Read ``num_streams`` shards concurrently, yielding their raw items interleaved.

    Reading shards one at a time makes training I/O-bound: the storage mount delivers only
    ~34 MB/s on a single stream but ~285 MB/s across 16, and a training step consumes ~41 MB
    of audio. Sequential reads therefore cost ~1.2 s/step — 90% of the step time — while the
    GPU sits idle. Each reader thread owns a disjoint slice of the shard list; tar reads
    release the GIL, so these genuinely overlap.
    """
    q: queue.Queue = queue.Queue(maxsize=num_streams * 64)
    stop = threading.Event()
    done = object()

    def reader(mine: list[str]) -> None:
        try:
            for p in mine:
                for item in _iter_shard_raw(p):
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                    if stop.is_set():
                        return
        finally:
            q.put(done)

    slices = [paths[i::num_streams] for i in range(num_streams)]
    slices = [s for s in slices if s]
    threads = [threading.Thread(target=reader, args=(s,), daemon=True) for s in slices]
    for t in threads:
        t.start()
    try:
        finished = 0
        while finished < len(threads):
            item = q.get()
            if item is done:
                finished += 1
                continue
            yield item
    finally:
        # A consumer that stops early (bucketing, max_steps) must not leave readers blocked
        # on a full queue forever.
        stop.set()
        while any(t.is_alive() for t in threads):
            try:
                q.get_nowait()
            except queue.Empty:
                pass


class ShardDataset(IterableDataset):
    def __init__(
        self,
        shards: list[str],
        tokenizer: PinyinTokenizer,
        sample_rate: int = 16000,
        augment: Callable[[Tensor], Tensor] | None = None,
        shuffle_buffer: int = 1000,
        shuffle_shards: bool = True,
        seed: int = 0,
        min_units: int = 1,
        infinite: bool = False,
        num_threads: int = 0,
        bucket_size: int = 0,
        batch_size: int = 0,
        read_streams: int = 0,
    ):
        super().__init__()
        self.shards = list(shards)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.augment = augment
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.min_units = min_units
        # >0: decode+augment in a thread pool (GIL-released work parallelizes across cores),
        # which is how we use all the cores without multiprocessing worker IPC.
        self.num_threads = num_threads
        # >0 (with batch_size): length bucketing — sort a pool by duration and emit batch-sized
        # groups, so batches are length-homogeneous and barely padded.
        self.bucket_size = bucket_size
        self.batch_size = batch_size
        # >1: read this many shards concurrently. The mount is ~34 MB/s single-stream but
        # ~285 MB/s at 16 streams, and sequential reads were 90% of the training step.
        self.read_streams = read_streams
        # infinite: cycle shards forever (reshuffling each pass). Under DDP this avoids the
        # epoch-boundary hang where unevenly-split shards give ranks different step counts and
        # desync the gradient all-reduce. Training is bounded by max_steps instead of epochs.
        self.infinite = infinite
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _my_shards(self, pass_idx: int = 0) -> list[str]:
        info = get_worker_info()
        worker = info.id if info else 0
        n_workers = info.num_workers if info else 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        else:
            rank, world = 0, 1
        shards = list(self.shards)
        if self.shuffle_shards:
            random.Random(self.seed + self.epoch + pass_idx * 7919).shuffle(shards)
        return select_shards(shards, rank, world, worker, n_workers)

    def _process(self, raw) -> dict | None:
        """Decode + tokenize + augment one (key, ext, audio_bytes, text)."""
        key, ext, audio_bytes, text = raw
        target = self.tokenizer.encode(text)
        if len(target) < self.min_units:
            return None
        wav = _decode_audio(audio_bytes, ext, self.sample_rate)
        if self.augment is not None:
            wav = self.augment(wav)
        return {"wav": wav, "target": torch.tensor(target, dtype=torch.long), "utt_id": key}

    def _samples(self, pass_idx: int = 0):
        my_shards = self._my_shards(pass_idx)
        if self.read_streams and self.read_streams > 1:
            raw_stream = _interleave_shards(my_shards, min(self.read_streams, len(my_shards)))
        else:
            raw_stream = (raw for shard in my_shards for raw in _iter_shard_raw(shard))
        if self.num_threads and self.num_threads > 1:
            yield from _threaded_map(self._process, raw_stream, self.num_threads)
        else:
            for raw in raw_stream:
                s = self._process(raw)
                if s is not None:
                    yield s

    def _all_samples(self):
        """One pass over this reader's shards, or endless reshuffled passes if infinite."""
        pass_idx = 0
        while True:
            yield from self._samples(pass_idx)
            if not self.infinite:
                return
            pass_idx += 1

    def _bucketed(self, rng):
        """Length bucketing: fill a pool, sort by duration, emit in batch-sized groups whose
        order is shuffled. Consecutive samples then have similar length, so the collated batch
        is padded to near the longest *in-bucket* clip instead of the longest in the corpus —
        far less wasted compute (and less padding polluting normalization statistics).
        """
        pool: list[dict] = []
        group = max(1, self.batch_size)
        for item in self._all_samples():
            pool.append(item)
            if len(pool) >= self.bucket_size:
                pool.sort(key=lambda x: x["wav"].numel())
                groups = [pool[i:i + group] for i in range(0, len(pool), group)]
                rng.shuffle(groups)            # keep batch order random across the epoch
                for g in groups:
                    yield from g
                pool = []
        if pool:
            pool.sort(key=lambda x: x["wav"].numel())
            groups = [pool[i:i + group] for i in range(0, len(pool), group)]
            rng.shuffle(groups)
            for g in groups:
                yield from g

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch + 1)
        if self.bucket_size and self.bucket_size > 1 and self.batch_size:
            yield from self._bucketed(rng)
            return
        if self.shuffle_buffer <= 1:
            yield from self._all_samples()
            return
        buf: list[dict] = []
        for item in self._all_samples():
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)  # only reached when not infinite
        yield from buf


def write_shards_from_manifest(
    manifest_path: str, out_dir: str, prefix: str = "shard",
    num_shards: int = 256, sample_rate: int = 16000, workers: int = 8, fmt: str = "flac",
) -> list[str]:
    return write_shards(read_manifest(manifest_path), out_dir, prefix, num_shards,
                        sample_rate, workers, fmt)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pack a manifest into WebDataset tar shards")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output directory for .tar shards")
    ap.add_argument("--prefix", default="shard")
    ap.add_argument("--num-shards", type=int, default=256)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--format", choices=["flac", "pcm"], default="flac",
                    help="pcm = raw int16, no decode at read time (fast training loop)")
    args = ap.parse_args()
    paths = write_shards_from_manifest(
        args.manifest, args.out, args.prefix, args.num_shards, args.sample_rate, args.workers,
        args.format,
    )
    print(f"wrote {len(paths)} {args.format} shards to {args.out}")


if __name__ == "__main__":
    main()
