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


def test_gpu_augment_keeps_lengths_consistent_with_audio(tmp_path):
    # _gpu_augment scatters differently-resampled sub-batches back into one padded tensor.
    # If the returned lengths don't track each clip's own factor, CTC's input_lengths stop
    # matching the audio -- so assert every reported length holds real (non-zero) samples.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.aug = AugConfig(enabled=True, specaug_enabled=False, gpu_speed=True, p_speed=1.0,
                        speed_factors=[0.9, 1.1], p_noise=0.0, p_rir=0.0, p_gain=0.0)
    trainer = Trainer(cfg)
    trainer.raw_model.train()

    torch.manual_seed(0)
    wavs = torch.zeros(6, 16000)
    lengths = torch.tensor([16000, 12000, 8000, 16000, 12000, 8000])
    for i, n in enumerate(lengths.tolist()):
        wavs[i, :n] = torch.randn(n) * 0.1

    out, out_len = trainer._gpu_augment(wavs, lengths)
    assert out.shape[0] == 6 and out_len.shape == (6,)
    assert int(out_len.max()) <= out.shape[-1]              # never point past the buffer
    for i in range(6):
        # each clip was stretched by one of the factors, in proportion to its own length
        ratio = out_len[i].item() / lengths[i].item()
        assert any(abs(ratio - r) < 0.02 for r in (1.0, 10 / 9, 10 / 11))
        assert out[i, : out_len[i]].abs().max() > 0         # audio really is that long
        assert out[i, out_len[i] + 40:].abs().max() < 1e-3  # and padding stayed padding


def test_gpu_augment_is_dtype_safe_under_autocast(tmp_path):
    # Resample/FFT-convolve are autocast-eligible, so under bf16 they return bf16 while the
    # scatter buffer is fp32 -- which crashed the first bf16 run. Augmentation must stay fp32.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.aug = AugConfig(enabled=True, specaug_enabled=False, gpu_speed=True, p_speed=1.0,
                        speed_factors=[0.9, 1.1], p_noise=0.0, p_rir=0.0, p_gain=0.0)
    trainer = Trainer(cfg)
    trainer.raw_model.train()

    dev = trainer.device                          # cuda when present, exactly as in training
    wavs = (torch.randn(4, 16000) * 0.1).to(dev)
    lengths = torch.tensor([16000, 12000, 16000, 8000])
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out, out_len = trainer._gpu_augment(wavs, lengths)
    assert out.dtype == torch.float32
    assert out.shape[0] == 4 and int(out_len.max()) <= out.shape[-1]


def test_ema_warmup_tracks_the_model_early(tmp_path):
    # Validation and best-checkpoint selection score the EMA weights. With a constant 0.999
    # decay the EMA is still ~14% random init at step 2000, so it reported an all-blank model
    # while the trained weights were emitting correctly. The decay must ramp in.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.ema_decay = 0.999
    trainer = Trainer(cfg)

    with torch.no_grad():
        for p in trainer.raw_model.parameters():
            p.fill_(1.0)
        for p in trainer.ema.parameters():
            p.fill_(0.0)
    for i in range(20):
        trainer.step = i
        trainer._update_ema()

    got = torch.stack([p.mean() for p in trainer.ema.parameters()]).mean().item()
    # a constant 0.999 decay would have moved only 1 - 0.999^20 = 2% of the way
    assert got > 0.75, f"EMA still {got:.3f} of the way to the model after 20 steps"


def test_prefetch_overlaps_loading_with_consumption():
    # The step was strictly serial -- fetch, compute, fetch -- so the GPU idled through every
    # fetch. Prefetching must let production and consumption proceed at the same time.
    import time

    from kws_mandarin.train.trainer import _prefetch

    def slow_source():
        for i in range(8):
            time.sleep(0.05)
            yield i

    t0 = time.perf_counter()
    for _ in _prefetch(slow_source(), depth=4):
        time.sleep(0.05)                       # "compute" as long as the "load"
    overlapped = time.perf_counter() - t0
    # serial would be 8*(0.05+0.05) = 0.8s; overlapped approaches 8*0.05 = 0.4s
    assert overlapped < 0.65, f"no overlap: {overlapped:.2f}s"


def test_prefetch_propagates_loader_errors():
    # A crash in the loader must reach the training loop. Swallowing it would leave the
    # consumer waiting on a queue that will never be filled -- a silent hang.
    import pytest

    from kws_mandarin.train.trainer import _prefetch

    def broken():
        yield 1
        raise RuntimeError("shard exploded")

    with pytest.raises(RuntimeError, match="shard exploded"):
        list(_prefetch(broken(), depth=2))


def test_prefetch_stops_producer_when_consumer_quits():
    # max_steps ends the loop mid-stream. The producer must not stay blocked on a full queue.
    import threading
    import time

    from kws_mandarin.train.trainer import _prefetch

    before = threading.active_count()
    gen = _prefetch(iter(range(10_000)), depth=2)
    next(gen)
    gen.close()
    for _ in range(100):
        if threading.active_count() <= before:
            break
        time.sleep(0.05)
    assert threading.active_count() <= before, "prefetch thread leaked"


def test_nonfinite_loss_is_skipped_not_backpropagated(tmp_path):
    # A single NaN loss propagates NaN into every weight and the run is dead from then on,
    # silently. The step must be skipped and the model left finite.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.optim.max_steps = 3
    cfg.val_every = 10_000
    trainer = Trainer(cfg)
    trainer._loss_from_batch = lambda batch: torch.tensor(float("nan"), requires_grad=True)

    trainer.train(resume=False)
    assert trainer.n_bad_loss == 3                        # every step was rejected
    assert all(torch.isfinite(p).all() for p in trainer.raw_model.parameters())


def test_checkpoint_round_trips_the_data_stream_position(tmp_path):
    # Resuming without the shard-stream position restarts at pass 0, re-training on exactly
    # the data the interrupted run already consumed. Only ShardDataset has this position.
    from kws_mandarin.data import write_shards

    train_m, dev_m = _corpus(tmp_path)
    from kws_mandarin.data.manifest import read_manifest
    shards = write_shards(read_manifest(train_m), str(tmp_path / "sh"), num_shards=2, workers=1)

    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.data.train_shards = str(tmp_path / "sh" / "*.tar")
    trainer = Trainer(cfg)
    assert trainer.train_dataset.pass_idx == 0
    trainer.train_dataset.pass_idx = 7
    trainer.save_checkpoint("latest.pt")

    fresh = Trainer(cfg)
    assert fresh.train_dataset.pass_idx == 0
    fresh.load_checkpoint(tmp_path / "ckpt" / "latest.pt")
    assert fresh.train_dataset.pass_idx == 7, "resumed run would replay data already trained on"


def test_final_validation_runs_on_every_rank(tmp_path):
    # validate_and_checkpoint ends in a collective barrier. Guarding the FINAL call with
    # is_main made rank 0 execute one more barrier than the others, so the job deadlocked
    # after the last step -- the log showed a finished run while the process hung for the
    # 30-minute process-group timeout. Non-main ranks must reach it too.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.optim.max_steps = 2
    cfg.val_every = 10_000                       # only the end-of-training call fires
    trainer = Trainer(cfg)

    calls = []
    trainer.validate_and_checkpoint = lambda: (calls.append(trainer.step), {})[1]
    trainer.is_main = False                      # pretend to be a non-main rank
    trainer.train(resume=False)

    assert calls, "non-main rank skipped the final validate_and_checkpoint -> barrier mismatch"


def test_accumulation_averages_gradients_rather_than_summing(tmp_path):
    # accum_steps lets a big model use a small micro-batch without changing the recipe:
    # global batch = batch_size * world * accum_steps. That requires dividing each micro-batch
    # loss by accum_steps -- summing instead would silently double the effective learning rate.
    #
    # Compared against per-micro-batch gradients, NOT against one large batch: the model has 25
    # BatchNorm layers whose statistics are per micro-batch, so accumulation is deliberately
    # not identical to a single larger batch. That is a real caveat of accum_steps, not a bug.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.data.batch_size = 2
    cfg.optim.accum_steps = 2
    cfg.optim.max_steps = 1
    cfg.val_every = 10_000
    cfg.optim.grad_clip = 1e9              # clipping would hide a factor-of-2 error
    trainer = Trainer(cfg)
    trainer.raw_model.eval()               # freeze BN running stats so both paths match exactly

    batches = []
    it = iter(trainer.loader)
    batches = [next(it), next(it)]

    def grad_of(batch):
        trainer.optimizer.zero_grad(set_to_none=True)
        trainer._loss_from_batch(batch).backward()
        return [p.grad.detach().clone() for p in trainer.raw_model.parameters() if p.grad is not None]

    g1, g2 = grad_of(batches[0]), grad_of(batches[1])
    expected = [(a + b) / 2 for a, b in zip(g1, g2)]

    trainer.optimizer.zero_grad(set_to_none=True)
    for b in batches:                      # exactly what the loop does with accum_steps=2
        (trainer._loss_from_batch(b) / 2).backward()
    actual = [p.grad.detach().clone() for p in trainer.raw_model.parameters() if p.grad is not None]

    for e, a in zip(expected, actual):
        assert torch.allclose(e, a, atol=1e-6), "accumulated gradient is not the mean"


def test_accumulation_takes_one_optimizer_step_per_cycle(tmp_path):
    # self.step must count OPTIMIZER steps, not micro-batches, or max_steps and the LR
    # schedule silently mean something different whenever accum_steps changes.
    train_m, dev_m = _corpus(tmp_path)
    cfg = _tiny_config(tmp_path, train_m, dev_m)
    cfg.data.batch_size = 2
    cfg.optim.accum_steps = 2
    cfg.optim.max_steps = 3
    cfg.val_every = 10_000
    trainer = Trainer(cfg)

    n_opt = []
    real = trainer.optimizer.step
    trainer.optimizer.step = lambda *a, **k: (n_opt.append(1), real(*a, **k))[1]
    trainer.train(resume=False)

    assert trainer.step == 3
    assert len(n_opt) == 3, f"expected 3 optimizer steps for 3 accumulated steps, got {len(n_opt)}"
