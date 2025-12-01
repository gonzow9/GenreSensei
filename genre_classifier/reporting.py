"""Metrics helpers and visualization utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_mapping: Sequence[str]
) -> Dict[str, object]:
    report = classification_report(
        y_true, y_pred, target_names=class_mapping, output_dict=True, zero_division=0
    )
    accuracy = float(report["accuracy"])
    return {"accuracy": accuracy, "report": report}


def save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_mapping: Sequence[str],
    path: Path,
) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_mapping))))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_mapping,
        yticklabels=class_mapping,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved confusion matrix plot -> %s", path)
    return path


def persist_metrics(metrics: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    save_json(metrics, path)
    logger.info("Saved metrics -> %s", path)
