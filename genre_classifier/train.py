"""Training orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from .config import AppConfig
from .data import load_cached_data, prepare_dataset_from_folder
from .models import train_cnn_model, train_svm_models, train_val_split
from .utils import set_global_seed

logger = logging.getLogger(__name__)


def run_training(config: AppConfig, train_target: str = "both", force_prepare: bool = False) -> Dict[str, Dict]:
    """
    Execute the full training pipeline.
    """
    set_global_seed(config.training.random_state)

    if force_prepare or not config.dataset.json_cache.exists():
        prepare_dataset_from_folder(config.dataset, overwrite=force_prepare)
    else:
        logger.info("Using existing cache at %s", config.dataset.json_cache)

    X, y, mapping = load_cached_data(config.dataset.json_cache)
    X_train, X_test, y_train, y_test = train_val_split(
        X.astype(np.float32), y, config
    )

    logger.info(
        "Dataset size: %d segments (%d train, %d test)",
        X.shape[0],
        X_train.shape[0],
        X_test.shape[0],
    )

    results: Dict[str, Dict] = {}

    if train_target in {"svm", "both"}:
        results["svm"] = train_svm_models(
            X_train, X_test, y_train, y_test, mapping, config
        )

    if train_target in {"cnn", "both"}:
        results["cnn"] = train_cnn_model(
            X_train, X_test, y_train, y_test, mapping, config
        )

    return results
