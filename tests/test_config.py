from kws_mandarin.config import TrainConfig


def test_yaml_roundtrip(tmp_path):
    cfg = TrainConfig()
    cfg.model.scale = 6.0
    cfg.optim.max_steps = 5000
    cfg.val_keywords = ["中国", "经济"]
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    loaded = TrainConfig.from_yaml(path)
    assert loaded.model.scale == 6.0
    assert loaded.optim.max_steps == 5000
    assert loaded.val_keywords == ["中国", "经济"]


def test_from_dict_partial_and_ignores_extras():
    cfg = TrainConfig.from_dict({
        "seed": 7,
        "model": {"scale": 4.0},              # partial nested override
        "unknown_top": 1,                     # ignored
        "optim": {"lr": 0.01, "bogus": 2},    # bogus ignored
    })
    assert cfg.seed == 7
    assert cfg.model.scale == 4.0
    assert cfg.model.n_mels == 40             # default preserved
    assert cfg.optim.lr == 0.01
