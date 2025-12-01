"""Prediction utilities for trained models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from .config import AppConfig
from .data import extract_mfcc_segments, load_cached_data
from .models import _flatten_features, _hybrid_features, _stat_features, load_svm_models
from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def _prepare_features_for_model(
    model_name: str, mfccs: np.ndarray
) -> np.ndarray:
    if model_name == "stats":
        return _stat_features(mfccs)
    if model_name == "flat":
        return _flatten_features(mfccs)
    if model_name == "hybrid":
        stats = _stat_features(mfccs)
        return _hybrid_features(mfccs, stats)
    raise ValueError(f"Unknown model name {model_name}")


def predict_audio(
    audio_path: Path, config: AppConfig, save_path: Path | None = None
) -> Dict[str, Tuple[str, float]]:
    _, _, class_mapping = load_cached_data(config.dataset.json_cache)
    mfccs = extract_mfcc_segments(audio_path, config.dataset)
    models = load_svm_models(config.artifacts.models_dir)

    if not models:
        raise FileNotFoundError("No trained SVM models found. Run train first.")

    results: Dict[str, Tuple[str, float]] = {}
    for name, model in models.items():
        features = _prepare_features_for_model(name, mfccs)
        predictions = model.predict(features)
        genre_votes: Dict[str, int] = {}
        for pred in predictions:
            genre = class_mapping[pred]
            genre_votes[genre] = genre_votes.get(genre, 0) + 1
        sorted_votes = sorted(genre_votes.items(), key=lambda x: x[1], reverse=True)
        top_genre, votes = sorted_votes[0]
        confidence = votes / len(predictions)
        logger.info("%s -> %s (%.2f)", name.upper(), top_genre, confidence)
        results[name] = (top_genre, confidence)

    if save_path:
        ensure_dir(save_path.parent)
        save_json(
            {
                "audio": str(audio_path),
                "results": {k: {"genre": v[0], "confidence": v[1]} for k, v in results.items()},
            },
            save_path,
        )
        logger.info("Saved prediction summary -> %s", save_path)
    return results
