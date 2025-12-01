"""Model training and loading utilities."""

from __future__ import annotations

import logging
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.utils import to_categorical

from .config import AppConfig
from .interpretability import compute_saliency_map, plot_saliency_map, rank_stat_features, save_feature_ranking
from .reporting import classification_metrics, persist_metrics, save_confusion_plot
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def _stat_features(X: np.ndarray) -> np.ndarray:
    return np.hstack(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            np.median(X, axis=1),
            np.percentile(X, 25, axis=1),
            np.percentile(X, 75, axis=1),
        ]
    )


def _flatten_features(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def _hybrid_features(X: np.ndarray, stats: np.ndarray) -> np.ndarray:
    sampled = X[:, ::5, :]
    sampled_flat = sampled.reshape(sampled.shape[0], -1)
    return np.hstack([stats, sampled_flat])


def train_val_split(
    X: np.ndarray, y: np.ndarray, config: AppConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=config.training.test_size,
        stratify=y if config.training.stratify else None,
        random_state=config.training.random_state,
    )


def train_svm_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_mapping: Sequence[str],
    config: AppConfig,
) -> Dict[str, Dict[str, object]]:
    models_dir = ensure_dir(config.artifacts.models_dir)

    X_train_stats = _stat_features(X_train)
    X_test_stats = _stat_features(X_test)

    X_train_flat = _flatten_features(X_train)
    X_test_flat = _flatten_features(X_test)

    X_train_hybrid = _hybrid_features(X_train, X_train_stats)
    X_test_hybrid = _hybrid_features(X_test, X_test_stats)

    logger.info("Training SVM (statistical features)")
    svm_stats = SVC(kernel=config.svm.kernel, C=config.svm.c_stats, gamma=config.svm.gamma, random_state=config.training.random_state)
    svm_stats.fit(X_train_stats, y_train)
    y_pred_stats = svm_stats.predict(X_test_stats)

    logger.info("Training SVM (flattened features)")
    svm_flat = SVC(kernel=config.svm.kernel, C=config.svm.c_flat, gamma=config.svm.gamma, random_state=config.training.random_state)
    svm_flat.fit(X_train_flat, y_train)
    y_pred_flat = svm_flat.predict(X_test_flat)

    logger.info("Training SVM (hybrid features)")
    svm_hybrid = SVC(kernel=config.svm.kernel, C=config.svm.c_hybrid, gamma=config.svm.gamma, random_state=config.training.random_state)
    svm_hybrid.fit(X_train_hybrid, y_train)
    y_pred_hybrid = svm_hybrid.predict(X_test_hybrid)

    for path, model in [
        (models_dir / "svm_stats.pkl", svm_stats),
        (models_dir / "svm_flat.pkl", svm_flat),
        (models_dir / "svm_hybrid.pkl", svm_hybrid),
    ]:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved model -> %s", path)

    reports_dir = ensure_dir(config.artifacts.reports_dir)
    metrics = {
        "stats": classification_metrics(y_test, y_pred_stats, class_mapping),
        "flat": classification_metrics(y_test, y_pred_flat, class_mapping),
        "hybrid": classification_metrics(y_test, y_pred_hybrid, class_mapping),
    }

    persist_metrics(metrics, reports_dir / "svm_metrics.json")
    save_confusion_plot(y_test, y_pred_stats, class_mapping, reports_dir / "svm_stats_confusion.png")

    ranking = rank_stat_features(X_train_stats, y_train, n_mfcc=X_train.shape[-1])
    save_feature_ranking(ranking, reports_dir / "svm_stat_feature_importance.json")

    return metrics


def build_cnn(input_shape: Sequence[int], num_classes: int, config: AppConfig) -> Sequential:
    filters = config.cnn.filters
    model = Sequential(
        [
            Conv2D(filters[0], (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            BatchNormalization(),
            Conv2D(filters[1], (3, 3), activation="relu"),
            MaxPooling2D((3, 3), strides=(2, 2), padding="same"),
            BatchNormalization(),
            Conv2D(filters[2], (2, 2), activation="relu"),
            MaxPooling2D((2, 2), strides=(2, 2), padding="same"),
            BatchNormalization(),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(config.cnn.dropout),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.cnn.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_mapping: Sequence[str],
    config: AppConfig,
) -> Dict[str, object]:
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]
    y_train_cnn = to_categorical(y_train)
    y_test_cnn = to_categorical(y_test)

    model = build_cnn(X_train_cnn.shape[1:], len(class_mapping), config)
    history = model.fit(
        X_train_cnn,
        y_train_cnn,
        validation_data=(X_test_cnn, y_test_cnn),
        epochs=config.cnn.epochs,
        batch_size=config.cnn.batch_size,
        verbose=2,
    )
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cnn, verbose=0)

    models_dir = ensure_dir(config.artifacts.models_dir)
    model_path = models_dir / "cnn.keras"
    model.save(model_path)
    logger.info("Saved CNN model -> %s", model_path)

    y_pred = np.argmax(model.predict(X_test_cnn, verbose=0), axis=1)

    reports_dir = ensure_dir(config.artifacts.reports_dir)
    metrics = {
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "classification": classification_metrics(y_test, y_pred, class_mapping),
    }
    persist_metrics(metrics, reports_dir / "cnn_metrics.json")
    save_confusion_plot(y_test, y_pred, class_mapping, reports_dir / "cnn_confusion.png")

    # Saliency on a single example
    saliency = compute_saliency_map(model, X_test_cnn[:1])
    plot_saliency_map(saliency, reports_dir / "cnn_saliency.png")

    return metrics


def load_svm_models(models_dir: Path) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for name in ["svm_stats", "svm_flat", "svm_hybrid"]:
        path = models_dir / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name.split("_")[1]] = pickle.load(f)
    return models
