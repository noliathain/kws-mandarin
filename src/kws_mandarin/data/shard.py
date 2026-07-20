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
import random
import tarfile
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

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


def _decode_audio(raw: bytes, sample_rate: int) -> Tensor:
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def _load_wav(path: str, sample_rate: int) -> Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data).mean(dim=1)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


# --- writing --------------------------------------------------------------------------

def _write_one_shard(args: tuple) -> tuple[str, int]:
    shard_path, utt_dicts, sample_rate = args
    n = 0
    with tarfile.open(shard_path, "w") as tar:
        for u in utt_dicts:
            try:
                wav = _load_wav(u["wav"], sample_rate)
            except Exception:
                continue
            key = u["utt_id"]
            audio = _encode_flac(wav, sample_rate)
            meta = json.dumps(
                {"text": u["text"], "duration": u["duration"], "speaker": u["speaker"]},
                ensure_ascii=False,
            ).encode("utf-8")
            for ext, payload in ((".flac", audio), (".json", meta)):
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
) -> list[str]:
    """Pack utterances into ``num_shards`` tar shards (round-robin for balance + shuffling)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    num_shards = max(1, min(num_shards, len(utterances)))
    # round-robin assignment balances shard sizes and pre-shuffles speakers across shards
    buckets: list[list[dict]] = [[] for _ in range(num_shards)]
    for i, u in enumerate(utterances):
        buckets[i % num_shards].append(asdict(u))

    tasks = [
        (str(out / f"{prefix}-{s:05d}.tar"), buckets[s], sample_rate)
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


def _assemble(key: str, parts: dict[str, bytes], sample_rate: int):
    meta = json.loads(parts["json"].decode("utf-8"))
    wav = _decode_audio(parts["flac"], sample_rate)
    return key, wav, meta["text"], meta["duration"]


def _iter_shard_raw(path: str):
    """Yield (key, flac_bytes, text) grouped by key — cheap sequential tar reads, NO decode.

    Decode/augment (the expensive, GIL-releasing part) is done downstream, in parallel threads.
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
                yield cur_key, cur["flac"], json.loads(cur["json"])["text"]
                cur = {}
            cur_key = key
            cur[ext] = data
        if cur_key is not None:
            yield cur_key, cur["flac"], json.loads(cur["json"])["text"]


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
        """Decode + tokenize + augment one (key, flac_bytes, text) — runs in the thread pool."""
        key, flac_bytes, text = raw
        target = self.tokenizer.encode(text)
        if len(target) < self.min_units:
            return None
        wav = _decode_audio(flac_bytes, self.sample_rate)
        if self.augment is not None:
            wav = self.augment(wav)
        return {"wav": wav, "target": torch.tensor(target, dtype=torch.long), "utt_id": key}

    def _samples(self, pass_idx: int = 0):
        raw_stream = (
            raw
            for shard in self._my_shards(pass_idx)
            for raw in _iter_shard_raw(shard)
        )
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

    def __iter__(self):
        if self.shuffle_buffer <= 1:
            yield from self._all_samples()
            return
        rng = random.Random(self.seed + self.epoch + 1)
        buf: list[dict] = []
        for item in self._all_samples():
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)  # only reached when not infinite
        yield from buf


def write_shards_from_manifest(
    manifest_path: str, out_dir: str, prefix: str = "shard",
    num_shards: int = 256, sample_rate: int = 16000, workers: int = 8,
) -> list[str]:
    return write_shards(read_manifest(manifest_path), out_dir, prefix, num_shards, sample_rate, workers)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Pack a manifest into WebDataset tar shards")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="output directory for .tar shards")
    ap.add_argument("--prefix", default="shard")
    ap.add_argument("--num-shards", type=int, default=256)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = write_shards_from_manifest(
        args.manifest, args.out, args.prefix, args.num_shards, args.sample_rate, args.workers
    )
    print(f"wrote {len(paths)} shards to {args.out}")


if __name__ == "__main__":
    main()
