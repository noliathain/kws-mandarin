"""CLI entry: python -m kws_mandarin.train --config configs/base.yaml

Under torchrun this launches one process per GPU:
    torchrun --standalone --nproc_per_node=8 -m kws_mandarin.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse

from ..config import TrainConfig
from .trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the KWS acoustic model")
    ap.add_argument("--config", required=True, help="path to a training YAML")
    ap.add_argument("--no-resume", action="store_true", help="ignore any existing latest.pt")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    trainer = Trainer(cfg)
    if trainer.is_main:
        print(f"model params: {trainer.raw_model.num_params():,} | vocab {trainer.tokenizer.vocab_size} "
              f"| world_size {trainer.world_size} | device {trainer.device}", flush=True)
        # State what augmentation actually loaded. A missing/None pack silently disables a
        # stage, and a run robust only on paper looks identical in the log to a real one.
        a = cfg.aug
        rir = f"{len(trainer.gpu_rirs)} RIRs" if trainer.gpu_rirs is not None else "OFF"
        noise = f"{len(trainer.gpu_noises)} clips" if trainer.gpu_noises is not None else "OFF"
        speed = f"{a.speed_factors}" if a.gpu_speed else "OFF"
        print(f"aug: rir={rir} (p={a.p_rir}) | noise={noise} (p={a.p_noise}, "
              f"snr={a.snr_db_min}-{a.snr_db_max}dB) | speed={speed} (p={a.p_speed}) | "
              f"specaug={'on' if a.specaug_enabled else 'OFF'}", flush=True)
    trainer.train(resume=not args.no_resume)


if __name__ == "__main__":
    main()
