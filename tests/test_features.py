import numpy as np
import soundfile as sf
from pathlib import Path

from genre_classifier.config import AppConfig
from genre_classifier.data import extract_mfcc_segments
from genre_classifier.models import _flatten_features, _hybrid_features, _stat_features


def test_stat_flat_hybrid_shapes():
    X = np.random.rand(4, 10, 13).astype(np.float32)
    stats = _stat_features(X)
    flat = _flatten_features(X)
    hybrid = _hybrid_features(X, stats)

    assert stats.shape == (4, 13 * 7)
    assert flat.shape == (4, 10 * 13)
    assert hybrid.shape[0] == 4
    assert hybrid.shape[1] == stats.shape[1] + (2 * 13)  # sampling every 5th frame from 10 -> 2


def test_extract_mfcc_segments(tmp_path: Path):
    config = AppConfig().dataset
    sr = config.sample_rate
    duration_sec = 1
    t = np.linspace(0, duration_sec, sr * duration_sec, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, signal, sr)

    mfccs = extract_mfcc_segments(audio_path, config)
    assert mfccs.ndim == 3
    assert mfccs.shape[-1] == config.n_mfcc
