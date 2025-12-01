"""Interpretability helpers: saliency for CNN and simple feature ranks for SVM stats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.feature_selection import f_classif

from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def compute_saliency_map(
    model: tf.keras.Model, input_batch: np.ndarray, class_index: int | None = None
) -> np.ndarray:
    """
    Compute a simple saliency map by taking the gradient of the target logit with
    respect to the input spectrogram.
    """
    input_tensor = tf.convert_to_tensor(input_batch)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = model(input_tensor, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        target = predictions[:, class_index]
    grads = tape.gradient(target, input_tensor)
    saliency = tf.reduce_mean(tf.abs(grads), axis=-1)
    return saliency.numpy()


def plot_saliency_map(
    saliency: np.ndarray, path: Path, title: str = "Saliency Map"
) -> Path:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(saliency[0], origin="lower", aspect="auto", cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("MFCC coefficients")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved saliency map -> %s", path)
    return path


def rank_stat_features(
    X_stats: np.ndarray, y: np.ndarray, n_mfcc: int
) -> List[Dict[str, float]]:
    """
    Rank statistical MFCC features via ANOVA F-score to highlight importance.
    """
    scores, _ = f_classif(X_stats, y)
    names = []
    stats = ["mean", "std", "min", "max", "median", "p25", "p75"]
    for stat in stats:
        for mfcc_idx in range(n_mfcc):
            names.append(f"{stat}_mfcc{mfcc_idx}")
    ranking = [
        {"feature": name, "score": float(score)}
        for name, score in sorted(zip(names, scores), key=lambda kv: kv[1], reverse=True)
    ]
    return ranking


def save_feature_ranking(ranking: List[Dict[str, float]], path: Path) -> None:
    save_json({"feature_ranking": ranking}, path)
    logger.info("Saved feature importance ranking -> %s", path)
