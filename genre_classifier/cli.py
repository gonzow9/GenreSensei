"""Command-line interface for the genre classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import AppConfig, load_config
from .data import prepare_dataset_from_folder, validate_dataset
from .predict import predict_audio
from .train import run_training
from .utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GTZAN Genre Classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Extract MFCCs and cache to JSON")
    prep.add_argument("--overwrite", action="store_true", help="Force re-extraction")

    train = subparsers.add_parser("train", help="Train SVM/CNN models")
    train.add_argument(
        "--target", choices=["svm", "cnn", "both"], default="both", help="Models to train"
    )
    train.add_argument(
        "--prepare", action="store_true", help="Prepare dataset cache before training"
    )

    predict = subparsers.add_parser("predict", help="Predict genre for an audio file")
    predict.add_argument("audio", type=Path, help="Path to audio file")
    predict.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save prediction JSON summary",
    )

    subparsers.add_parser("validate", help="Validate dataset structure")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)
    config: AppConfig = load_config(args.config)

    if args.command == "prepare":
        prepare_dataset_from_folder(config.dataset, overwrite=args.overwrite)
        return

    if args.command == "train":
        run_training(config, train_target=args.target, force_prepare=args.prepare)
        return

    if args.command == "predict":
        predict_audio(args.audio, config, save_path=args.save)
        return

    if args.command == "validate":
        validate_dataset(config.dataset.root)
        return


if __name__ == "__main__":
    main()
