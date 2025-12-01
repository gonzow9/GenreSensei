from genre_classifier.config import AppConfig, load_config


def test_load_default_config():
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.dataset.sample_rate == 22050
    assert cfg.training.test_size == 0.25
