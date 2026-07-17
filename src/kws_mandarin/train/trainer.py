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
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from ..config import TrainConfig
from ..data import KWSDataset, SpecAugment, WaveformAugment, collate_kws
from ..data.manifest import read_manifest
from ..loss import CTCLoss
from ..models import KWSModel
from ..tokenizer import PinyinTokenizer, ToneMode
from .scheduler import warmup_cosine_scheduler
from .validate_kws import run_validation


class Trainer:
    def __init__(self, config: TrainConfig, tokenizer=None, train_dataset=None, dev_utts=None):
        self.cfg = config
        self._setup_distributed()
        self._set_seed(config.seed + self.rank)

        self.tokenizer = tokenizer or PinyinTokenizer(ToneMode(config.model.tone_mode))
        self._build_model()
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
                dist.init_process_group(backend=backend)
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
        return WaveformAugment.from_dirs(
            a.musan_dir, a.rir_dir, sample_rate=self.cfg.data.sample_rate,
            snr_db_range=(a.snr_db_min, a.snr_db_max),
            p_noise=a.p_noise, p_rir=a.p_rir, p_speed=a.p_speed, p_gain=a.p_gain,
            seed=self.cfg.seed + self.rank,
        )

    def _build_data(self, train_dataset, dev_utts) -> None:
        if train_dataset is None:
            train_dataset = KWSDataset(
                self.cfg.data.train_manifest, self.tokenizer,
                sample_rate=self.cfg.data.sample_rate, augment=self._make_augment(),
            )
        self.train_dataset = train_dataset

        sampler = DistributedSampler(train_dataset) if self.distributed else None
        self.sampler = sampler
        self.loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate_kws,
            drop_last=True,
            pin_memory=self.use_cuda,
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

    def _loss_from_batch(self, batch) -> torch.Tensor:
        wavs = batch["wavs"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        input_lengths = self.raw_model.output_lengths(batch["wav_lengths"])  # CPU
        logits = self.model(wavs)
        return self.criterion(logits, targets, input_lengths, batch["target_lengths"])

    def _update_ema(self) -> None:
        if self.ema is None:
            return
        d = self.cfg.ema_decay
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
