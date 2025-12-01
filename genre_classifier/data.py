"""Data preparation, feature extraction, and validation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np

from .config import DatasetConfig
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def validate_dataset(dataset_path: Path) -> Dict[str, int]:
    """
    Validate the GTZAN folder structure and return a per-genre file count.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    genre_counts: Dict[str, int] = {}
    for genre_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        wavs = list(genre_dir.glob("*.wav"))
        genre_counts[genre_dir.name] = len(wavs)

    logger.info("Found %d genres: %s", len(genre_counts), list(genre_counts.keys()))
    return genre_counts


def prepare_dataset_from_folder(config: DatasetConfig, overwrite: bool = False) -> None:
    """
    Scan *.wav, extract MFCCs, and cache to json.
    """
    json_path = config.json_cache
    if json_path.exists() and not overwrite:
        logger.info("MFCC cache already exists -> %s", json_path)
        return

    counts = validate_dataset(config.root)
    data = {"mapping": sorted(counts.keys()), "mfccs": [], "labels": []}

    samples_per_track = config.sample_rate * config.duration
    samples_per_segment = int(samples_per_track / config.num_segments)
    expected_vectors = int(np.ceil(samples_per_segment / config.hop_length))

    ensure_dir(json_path.parent)

    logger.info("Extracting MFCCs (n_mfcc=%d). This can take a few minutes.", config.n_mfcc)
    for label, genre in enumerate(data["mapping"]):
        wav_paths = list((config.root / genre).glob("*.wav"))
        if config.max_files_per_genre:
            wav_paths = wav_paths[: config.max_files_per_genre]
        for wav_path in wav_paths:
            try:
                signal, _ = librosa.load(str(wav_path), sr=config.sample_rate)
            except Exception as exc:  # pragma: no cover - depends on filesystem state
                logger.warning("Skipping corrupted file %s (%s)", wav_path, exc)
                continue

            if signal.shape[0] < samples_per_track:
                signal = np.pad(signal, (0, samples_per_track - signal.shape[0]))
            else:
                signal = signal[:samples_per_track]

            for segment in range(config.num_segments):
                start = samples_per_segment * segment
                end = start + samples_per_segment
                mfcc = librosa.feature.mfcc(
                    y=signal[start:end],
                    sr=config.sample_rate,
                    n_mfcc=config.n_mfcc,
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                ).T

                if mfcc.shape[0] == expected_vectors:
                    data["mfccs"].append(mfcc.tolist())
                    data["labels"].append(label)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved extracted data -> %s", json_path)


def load_cached_data(json_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    X = np.array(data["mfccs"], dtype=np.float32)
    y = np.array(data["labels"])
    return X, y, data["mapping"]


def extract_mfcc_segments(audio_path: Path, config: DatasetConfig) -> np.ndarray:
    """
    Extract MFCC features from an audio file, handling files longer than 30 seconds by
    segmenting them into 30-second chunks.
    """
    logger.info("Loading audio file: %s", audio_path)
    samples_per_track = config.sample_rate * config.duration
    samples_per_segment = int(samples_per_track / config.num_segments)
    expected_vectors = int(np.ceil(samples_per_segment / config.hop_length))

    signal, _ = librosa.load(str(audio_path), sr=config.sample_rate)
    segments_30s = len(signal) // samples_per_track
    if segments_30s == 0:
        logger.info("Audio shorter than 30s, padding to %d samples", samples_per_track)
        signal = np.pad(signal, (0, samples_per_track - len(signal)))
        segments_30s = 1

    all_mfccs = []
    for chunk_idx in range(segments_30s):
        start_chunk = chunk_idx * samples_per_track
        end_chunk = start_chunk + samples_per_track
        chunk_signal = signal[start_chunk:end_chunk]

        for segment in range(config.num_segments):
            start = samples_per_segment * segment
            end = start + samples_per_segment
            mfcc = librosa.feature.mfcc(
                y=chunk_signal[start:end],
                sr=config.sample_rate,
                n_mfcc=config.n_mfcc,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
            ).T
            if mfcc.shape[0] == expected_vectors:
                all_mfccs.append(mfcc)

    return np.array(all_mfccs, dtype=np.float32)
