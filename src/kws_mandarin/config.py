"""Typed, YAML-backed training configuration.

One config object fully determines a run (reproducibility). Nested dataclasses map 1:1 to
sections in the YAML. ``TrainConfig.from_yaml`` / ``to_yaml`` round-trip losslessly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    scale: float = 3.0
    n_mels: int = 40
    causal: bool = True
    ssn_bands: int = 5
    dropout: float = 0.1
    tone_mode: str = "final"  # final | separate | syllable


@dataclass
class AugConfig:
    enabled: bool = True
    musan_dir: str | None = None
    rir_dir: str | None = None
    snr_db_min: float = 0.0
    snr_db_max: float = 20.0
    p_noise: float = 0.6
    p_rir: float = 0.5
    p_speed: float = 0.5
    p_gain: float = 0.5
    # SpecAugment (feature-domain)
    specaug_enabled: bool = True
    specaug_freq_mask: int = 8
    specaug_n_freq: int = 2
    specaug_time_mask: int = 25
    specaug_n_time: int = 2


@dataclass
class OptimConfig:
    lr: float = 3e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.98
    grad_clip: float = 5.0
    warmup_steps: int = 2000
    max_steps: int = 120000
    min_lr_ratio: float = 0.01


@dataclass
class DataConfig:
    train_manifest: str = "/teamspace/lightning_storage/kws-mandarin/manifests/aishell1_train.jsonl"
    dev_manifest: str = "/teamspace/lightning_storage/kws-mandarin/manifests/aishell1_dev.jsonl"
    sample_rate: int = 16000
    batch_size: int = 128
    num_workers: int = 8
    max_duration_s: float = 16.0  # drop utterances longer than this


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    seed: int = 1337
    precision: str = "bf16"       # bf16 | fp16 | fp32
    ema_decay: float = 0.0         # 0 disables EMA
    log_every: int = 50
    val_every: int = 2000
    ckpt_dir: str = "/teamspace/lightning_storage/kws-mandarin/checkpoints/base"
    keep_last: int = 3
    # keywords (Han strings) used for FRR@FAH validation during training
    val_keywords: list[str] = field(default_factory=list)
    val_max_utts: int = 2000       # cap dev utts scored per validation for speed

    # -- (de)serialization -------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        return _from_dict(cls, d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


def _from_dict(cls, d: dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a plain dict, ignoring extras.

    Uses ``get_type_hints`` so nested dataclass fields resolve to real classes even under
    ``from __future__ import annotations`` (where ``field.type`` is a string).
    """
    if not is_dataclass(cls):
        return d
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        ftype = hints.get(f.name)
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)
