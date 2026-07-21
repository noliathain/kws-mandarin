"""DDP trainer for the KWS acoustic model.

Runs single-process on CPU (tests) and scales to N GPUs under ``torchrun`` unchanged — DDP is
enabled only when ``WORLD_SIZE > 1``. Includes AMP (bf16/fp16), gradient clipping, warmup+
cosine LR, optional weight EMA, checkpoint/resume, and periodic FRR@FAH validation.

Robustness is wired in here: ``WaveformAugment`` (MUSAN/RIR/speed/gain) on the dataset and
``SpecAugment`` inside the model — both active only in training mode.

Launch (8 GPUs):
    torchrun --standalone --nproc_per_node=8 -m kws_mandarin.train --config configs/base.yaml
"""

from __future__ import annotations

import copy
import glob
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from ..config import TrainConfig
from ..data import (
    KWSDataset,
    ShardDataset,
    SpecAugment,
    WaveformAugment,
    add_noise_batch,
    apply_rir_batch,
    collate_kws,
    speed_perturb_batch,
)
from ..data.manifest import read_manifest
from ..loss import CTCLoss
from ..models import KWSModel
from ..tokenizer import PinyinTokenizer, ToneMode
from .scheduler import warmup_cosine_scheduler
from .validate_kws import run_validation

# Filesystem-based tensor sharing avoids PyTorch's file-descriptor sharing race
# ("could not unlink the shared memory file") that hangs the DataLoader under load with large
# batches + persistent workers — and, under DDP, hangs the whole job (a stalled worker stops
# one rank calling backward(), so every other rank blocks in the gradient all-reduce).
torch.multiprocessing.set_sharing_strategy("file_system")


def _dl_worker_init(_worker_id: int) -> None:
    """Run in every DataLoader worker: use filesystem tensor sharing there too."""
    torch.multiprocessing.set_sharing_strategy("file_system")


class Trainer:
    def __init__(self, config: TrainConfig, tokenizer=None, train_dataset=None, dev_utts=None):
        self.cfg = config
        self._setup_distributed()
        self._set_seed(config.seed + self.rank)

        self.tokenizer = tokenizer or PinyinTokenizer(ToneMode(config.model.tone_mode))
        self._build_model()
        self._build_gpu_rirs()
        self._build_data(train_dataset, dev_utts)
        self._build_optim()

        self.step = 0
        self.best_metric = float("inf")

    # -- setup ---------------------------------------------------------------------------

    def _setup_distributed(self) -> None:
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        self.use_cuda = torch.cuda.is_available()
        if self.distributed:
            backend = "nccl" if self.use_cuda else "gloo"
            if not dist.is_initialized():
                # Generous timeout so a long rank-0 validation doesn't trip the collective
                # watchdog while the other ranks wait at the barrier.
                dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        if self.use_cuda:
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.is_main = self.rank == 0

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(seed)

    def _amp_dtype(self):
        if not self.use_cuda or self.cfg.precision == "fp32":
            return None
        return torch.bfloat16 if self.cfg.precision == "bf16" else torch.float16

    def _build_model(self) -> None:
        spec = None
        a = self.cfg.aug
        if a.enabled and a.specaug_enabled:
            spec = SpecAugment(a.specaug_freq_mask, a.specaug_n_freq, a.specaug_time_mask, a.specaug_n_time)
        model = KWSModel(
            vocab_size=self.tokenizer.vocab_size,
            n_mels=self.cfg.model.n_mels,
            scale=self.cfg.model.scale,
            causal=self.cfg.model.causal,
            ssn_bands=self.cfg.model.ssn_bands,
            dropout=self.cfg.model.dropout,
            blank_id=self.tokenizer.blank_id,
            spec_augment=spec,
        ).to(self.device)
        self.raw_model = model
        if self.distributed:
            self.model = DDP(model, device_ids=[self.local_rank] if self.use_cuda else None)
        else:
            self.model = model
        self.ema = copy.deepcopy(model) if self.cfg.ema_decay > 0 else None
        if self.ema is not None:
            for p in self.ema.parameters():
                p.requires_grad_(False)

    def _make_augment(self):
        a = self.cfg.aug
        if not a.enabled:
            return None
        # Stages that run on the GPU are disabled here so they aren't also applied per-clip on CPU.
        return WaveformAugment.from_dirs(
            (None if a.gpu_noise else a.musan_dir), a.rir_dir,
            sample_rate=self.cfg.data.sample_rate,
            rir_pack=(None if a.gpu_rir else a.rir_pack),
            noise_pack=(None if a.gpu_noise else a.noise_pack),
            snr_db_range=(a.snr_db_min, a.snr_db_max),
            p_noise=(0.0 if a.gpu_noise else a.p_noise),
            p_rir=(0.0 if a.gpu_rir else a.p_rir),
            p_speed=(0.0 if a.gpu_speed else a.p_speed), p_gain=a.p_gain,
            seed=self.cfg.seed + self.rank,
        )

    def _build_gpu_rirs(self) -> None:
        """Load RIR / noise packs onto the device as dense banks for batched GPU augmentation."""
        a = self.cfg.aug
        sr = self.cfg.data.sample_rate
        self.gpu_rirs = None
        self.gpu_noises = None
        if not a.enabled:
            return
        if a.gpu_rir and a.rir_pack:
            from ..data import load_rir_pack

            rirs = load_rir_pack(a.rir_pack, sample_rate=sr)
            lmax = max(r.numel() for r in rirs)
            padded = torch.zeros(len(rirs), lmax)
            for i, r in enumerate(rirs):
                padded[i, : r.numel()] = r
            self.gpu_rirs = padded.to(self.device)
        if a.gpu_noise and a.noise_pack:
            from ..data import load_noise_pack

            noises = load_noise_pack(a.noise_pack, sample_rate=sr)
            # Tile every clip to a common length >= any batch, so a random crop always exists.
            length = int(self.cfg.data.max_duration_s * sr)
            bank = torch.zeros(len(noises), length)
            for i, n in enumerate(noises):
                if n.numel() == 0:
                    continue
                reps = (length + n.numel() - 1) // n.numel()
                bank[i] = n.repeat(reps)[:length]
            self.gpu_noises = bank.to(self.device)

    def _make_train_dataset(self):
        d = self.cfg.data
        augment = self._make_augment()
        if d.train_shards:
            shards = sorted(glob.glob(d.train_shards))
            if not shards:  # allow passing a directory
                shards = sorted(glob.glob(str(Path(d.train_shards) / "*.tar")))
            if not shards:
                raise FileNotFoundError(f"no shards matched: {d.train_shards}")
            return ShardDataset(
                shards, self.tokenizer, sample_rate=d.sample_rate, augment=augment,
                shuffle_buffer=d.shuffle_buffer, seed=self.cfg.seed,
                # cycle forever: training is bounded by max_steps, and under DDP this prevents
                # the epoch-boundary desync/hang from unevenly-split shards across ranks.
                infinite=self.distributed,
                num_threads=d.loader_threads,
                bucket_size=d.bucket_size,
                batch_size=d.batch_size,
            )
        return KWSDataset(d.train_manifest, self.tokenizer, sample_rate=d.sample_rate, augment=augment)

    def _build_data(self, train_dataset, dev_utts) -> None:
        if train_dataset is None:
            train_dataset = self._make_train_dataset()
        self.train_dataset = train_dataset

        # IterableDataset (shards) partitions internally — no DistributedSampler, no shuffle flag.
        self.iterable = isinstance(train_dataset, IterableDataset)
        sampler = DistributedSampler(train_dataset) if (self.distributed and not self.iterable) else None
        self.sampler = sampler
        nw = self.cfg.data.num_workers
        # forkserver: workers fork from a clean server process, NOT from the main process that
        # has already initialized CUDA/OpenMP threads (a plain fork there deadlocks the workers
        # on inherited locks). Requires the dataset + augmentation to be picklable.
        mp_context = "forkserver" if (nw > 0 and self.use_cuda) else None
        self.loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=(sampler is None and not self.iterable),
            sampler=sampler,
            num_workers=nw,
            collate_fn=collate_kws,
            drop_last=True,
            pin_memory=self.use_cuda,
            # keep the (audio-decode + augmentation) pipeline ahead of the tiny model so the
            # GPU is not starved: don't respawn workers each epoch, and prefetch several batches.
            persistent_workers=nw > 0,
            prefetch_factor=(self.cfg.data.prefetch_factor if nw > 0 else None),
            multiprocessing_context=mp_context,
            worker_init_fn=(_dl_worker_init if nw > 0 else None),
        )
        # dev utterances for validation (main process only)
        if dev_utts is None and self.is_main and Path(self.cfg.data.dev_manifest).exists():
            dev_utts = read_manifest(self.cfg.data.dev_manifest)
        self.dev_utts = dev_utts

    def _build_optim(self) -> None:
        o = self.cfg.optim
        self.optimizer = AdamW(
            self.model.parameters(), lr=o.lr, weight_decay=o.weight_decay, betas=(o.beta1, o.beta2)
        )
        self.scheduler = warmup_cosine_scheduler(self.optimizer, o.warmup_steps, o.max_steps, o.min_lr_ratio)
        self.criterion = CTCLoss(blank=self.tokenizer.blank_id)
        self.scaler = torch.amp.GradScaler(
            "cuda" if self.use_cuda else "cpu",
            enabled=(self._amp_dtype() == torch.float16),
        )

    # -- training ------------------------------------------------------------------------

    def _gpu_augment(self, wavs: torch.Tensor, lengths: torch.Tensor):
        """Run the augmentation chain outside autocast, in fp32.

        Resampling and FFT convolution are autocast-eligible, so under bf16 they would return
        bf16 (breaking the fp32 scatter buffer) and compute SNR scaling at ~3 decimal digits.
        Augmentation is signal processing on the input, not part of the model's math.
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self._augment_waveforms(wavs.float(), lengths)

    def _augment_waveforms(self, wavs: torch.Tensor, lengths: torch.Tensor):
        """Batched GPU augmentation, in physically-correct order for far-field robustness:
        change the talker's speaking rate (speed), reverberate that in a room (RIR), THEN add
        the room's ambient noise at a sampled SNR. Each stage hits an independent random
        fraction of the batch. Returns (wavs, lengths) — speed perturbation changes duration.
        """
        if not self.raw_model.training:
            return wavs, lengths
        a = self.cfg.aug
        b, t = wavs.shape
        dev = wavs.device
        lengths = lengths.to(dev)

        if a.gpu_speed and a.speed_factors and a.p_speed > 0:
            # Kaldi-style discrete speed perturbation: one batched resample per factor, rather
            # than a per-clip resample in the loader (which cost ~0.6 ms/utt single-threaded).
            factors = [1.0] + list(a.speed_factors)
            pick = torch.randint(0, len(factors), (b,), device=dev)
            pick = torch.where(torch.rand(b, device=dev) < a.p_speed, pick,
                               torch.zeros_like(pick))          # (1-p_speed) stay at 1.0x
            parts = []
            for fi, f in enumerate(factors):
                sel = (pick == fi).nonzero().flatten()
                if sel.numel():
                    parts.append((sel, *speed_perturb_batch(wavs[sel], lengths[sel], f)))
            tmax = max(p[1].shape[-1] for p in parts)
            out = wavs.new_zeros(b, tmax)
            new_lengths = torch.empty_like(lengths)
            for sel, w, ln in parts:
                out[sel, : w.shape[-1]] = w
                new_lengths[sel] = ln
            wavs, lengths, t = out, new_lengths, tmax

        if self.gpu_rirs is not None and a.p_rir > 0:
            hit = torch.rand(b, device=dev) < a.p_rir
            if bool(hit.any()):
                idx = torch.randint(0, self.gpu_rirs.shape[0], (b,), device=dev)
                wavs = torch.where(hit.unsqueeze(1), apply_rir_batch(wavs, self.gpu_rirs[idx]), wavs)

        if self.gpu_noises is not None and a.p_noise > 0:
            hit = torch.rand(b, device=dev) < a.p_noise
            if bool(hit.any()):
                bank_n, bank_len = self.gpu_noises.shape
                idx = torch.randint(0, bank_n, (b,), device=dev)
                max_off = max(1, bank_len - t + 1)
                off = torch.randint(0, max_off, (b,), device=dev)
                pos = (off.unsqueeze(1) + torch.arange(t, device=dev)).clamp(max=bank_len - 1)
                noise = self.gpu_noises[idx].gather(1, pos)          # (B, T) random crops
                snr = torch.empty(b, device=dev).uniform_(a.snr_db_min, a.snr_db_max)
                noisy = add_noise_batch(wavs, noise, snr, lengths)
                wavs = torch.where(hit.unsqueeze(1), noisy, wavs)
        return wavs, lengths

    def _loss_from_batch(self, batch) -> torch.Tensor:
        wavs = batch["wavs"].to(self.device, non_blocking=True)
        # augmentation may change duration (speed), so lengths come back from it
        wavs, wav_lengths = self._gpu_augment(wavs, batch["wav_lengths"])
        targets = batch["targets"].to(self.device, non_blocking=True)
        input_lengths = self.raw_model.output_lengths(wav_lengths.cpu())  # CTC wants CPU
        logits = self.model(wavs, wav_lengths)
        return self.criterion(logits, targets, input_lengths, batch["target_lengths"])

    def _update_ema(self) -> None:
        if self.ema is None:
            return
        # Ramp the decay in. A constant 0.999 still carries 0.999^2000 ~ 14% of the RANDOM INIT
        # at step 2000, and averages over the pre-alignment trajectory -- so early validation
        # scored a degenerate all-blank model that training had long since moved past.
        d = min(self.cfg.ema_decay, (1.0 + self.step) / (10.0 + self.step))
        with torch.no_grad():
            for e, m in zip(self.ema.parameters(), self.raw_model.parameters()):
                e.mul_(d).add_(m, alpha=1 - d)
            for e, m in zip(self.ema.buffers(), self.raw_model.buffers()):
                e.copy_(m)

    def train(self, resume: bool = True) -> None:
        if resume:
            self._maybe_resume()
        self.model.train()
        amp_dtype = self._amp_dtype()
        max_steps = self.cfg.optim.max_steps
        epoch = 0
        t0 = time.time()
        while self.step < max_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            elif hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)  # reshuffle shards each epoch
            for batch in self.loader:
                if self.step >= max_steps:
                    break
                self.optimizer.zero_grad(set_to_none=True)
                if amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        loss = self._loss_from_batch(batch)
                else:
                    loss = self._loss_from_batch(batch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self._update_ema()

                self.step += 1
                if self.is_main and self.step % self.cfg.log_every == 0:
                    rate = self.cfg.log_every / (time.time() - t0)
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"step {self.step}/{max_steps} loss {loss.item():.4f} "
                          f"lr {lr:.2e} {rate:.1f} it/s", flush=True)
                    t0 = time.time()

                if self.step % self.cfg.val_every == 0:
                    self.validate_and_checkpoint()
                    self.model.train()
            epoch += 1
        if self.is_main:
            self.validate_and_checkpoint()
        if self.distributed:
            dist.barrier()

    # -- validation / checkpointing ------------------------------------------------------

    @torch.no_grad()
    def validate_and_checkpoint(self) -> dict:
        metrics: dict = {}
        if self.is_main and self.dev_utts:
            eval_model = self.ema if self.ema is not None else self.raw_model
            metrics = run_validation(
                eval_model, self.dev_utts, self.tokenizer, self.cfg.val_keywords,
                self.device, sample_rate=self.cfg.data.sample_rate, max_utts=self.cfg.val_max_utts,
                use_ntc=self.cfg.val_use_ntc, ntc_lambda=self.cfg.ntc_lambda,
            )
            print(f"[val] step {self.step} " +
                  " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            key = metrics.get("frr@1.0", metrics.get("ter", float("inf")))
            self.save_checkpoint("latest.pt")
            if key < self.best_metric:
                self.best_metric = key
                self.save_checkpoint("best.pt")
        if self.distributed:
            dist.barrier()
        return metrics

    def _state(self) -> dict:
        return {
            "step": self.step,
            "best_metric": self.best_metric,
            "model": self.raw_model.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.cfg.to_dict(),
        }

    def save_checkpoint(self, name: str) -> None:
        d = Path(self.cfg.ckpt_dir)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self._state(), d / name)

    def _maybe_resume(self) -> None:
        latest = Path(self.cfg.ckpt_dir) / "latest.pt"
        if latest.exists():
            self.load_checkpoint(latest)
            if self.is_main:
                print(f"resumed from {latest} at step {self.step}", flush=True)

    def load_checkpoint(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.raw_model.load_state_dict(ckpt["model"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.best_metric = ckpt["best_metric"]
