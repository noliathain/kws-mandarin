import soundfile as sf
import torch

from kws_mandarin.config import AugConfig, DataConfig, ModelConfig, OptimConfig, TrainConfig
from kws_mandarin.data import Utterance, write_manifest
from kws_mandarin.train import Trainer


def _wav(tmp_path, uid, dur):
    p = tmp_path / f"{uid}.wav"
    sf.write(str(p), (torch.randn(int(dur * 16000)) * 0.1).numpy(), 16000)
    return str(p)


def _corpus(tmp_path):
    train = [
        Utterance("t1", _wav(tmp_path, "t1", 1.2), "你好世界", 1.2, "S1", "train"),
        Utterance("t2", _wav(tmp_path, "t2", 1.0), "打开空调", 1.0, "S2", "train"),
        Utterance("t3", _wav(tmp_path, "t3", 1.1), "播放音乐", 1.1, "S3", "train"),
        Utterance("t4", _wav(tmp_path, "t4", 0.9), "你好朋友", 0.9, "S4", "train"),
    ]
    dev = [
        Utterance("d1", _wav(tmp_path, "d1", 1.0), "你好世界", 1.0, "S5", "dev"),   # contains 你好
        Utterance("d2", _wav(tmp_path, "d2", 1.0), "关闭电视", 1.0, "S6", "dev"),   # does not
    ]
    tm = tmp_path / "train.jsonl"
    dm = tmp_path / "dev.jsonl"
    write_manifest(tm, train)
    write_manifest(dm, dev)
    return str(tm), str(dm)


def _tiny_config(tmp_path, train_m, dev_m):
    return TrainConfig(
        model=ModelConfig(scale=1.5, dropout=0.0),
        data=DataConfig(train_manifest=train_m, dev_manifest=dev_m, batch_size=2, num_workers=0),
        aug=AugConfig(enabled=False, specaug_enabled=False),
        optim=OptimConfig(lr=1e-3, warmup_steps=1, max_steps=4, min_lr_ratio=0.1),
        precision="fp32", ema_decay=0.0, log_every=1, val_every=2,
        ckpt_dir=str(tmp_path / "ckpt"), val_keywords=["你好"], val_max_utts=10,
    )


def test_end_to_end_training_and_checkpoint(tmp_path):
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    trainer = Trainer(cfg)
    trainer.train(resume=False)

    assert trainer.step == 4
    assert (tmp_path / "ckpt" / "latest.pt").exists()
    assert (tmp_path / "ckpt" / "best.pt").exists()


def test_validation_reports_metrics(tmp_path):
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    trainer = Trainer(cfg)
    metrics = trainer.validate_and_checkpoint()
    assert "ter" in metrics                       # token error rate always present
    assert "frr@1.0" in metrics                   # dev has both a positive and a negative
    assert 0.0 <= metrics["ter"]


def test_resume_restores_step(tmp_path):
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    Trainer(cfg).train(resume=False)              # produces latest.pt at step 4

    fresh = Trainer(cfg)
    assert fresh.step == 0
    fresh.load_checkpoint(tmp_path / "ckpt" / "latest.pt")
    assert fresh.step == 4
    assert fresh.best_metric < float("inf")


def test_loss_decreases_on_overfit(tmp_path):
    # A tiny model should drive CTC loss down when overfitting 4 clips over many steps.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.optim.max_steps = 60
    cfg.optim.warmup_steps = 5
    cfg.val_every = 10_000  # skip validation during this run
    trainer = Trainer(cfg)

    batch = next(iter(trainer.loader))
    first = trainer._loss_from_batch(batch).item()
    trainer.train(resume=False)
    last = trainer._loss_from_batch(batch).item()
    assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"
