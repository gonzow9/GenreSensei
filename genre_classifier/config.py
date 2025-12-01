"""Configuration loading and validation for the genre classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _resolve_path(value: Path | str) -> Path:
    """Resolve paths relative to the project root."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class DatasetConfig(BaseModel):
    root: Path = Field(default=Path("Data 2/genres_original"))
    json_cache: Path = Field(default=Path("data_gtzan.json"))
    sample_rate: int = 22050
    duration: int = 30
    num_segments: int = 5
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    max_files_per_genre: Optional[int] = None

    # pylint: disable=no-self-argument
    @validator("root", "json_cache", pre=True)
    def _resolve(cls, value: Path | str) -> Path:  # type: ignore[misc]
        return _resolve_path(value)


class TrainingConfig(BaseModel):
    test_size: float = 0.25
    random_state: int = 10
    stratify: bool = True


class SVMConfig(BaseModel):
    c_stats: float = 100.0
    c_flat: float = 1.0
    c_hybrid: float = 10.0
    kernel: str = "rbf"
    gamma: str = "scale"


class CNNConfig(BaseModel):
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    dropout: float = 0.3
    filters: tuple[int, int, int] = (32, 32, 32)


class ArtifactsConfig(BaseModel):
    base_dir: Path = Field(default=Path("artifacts"))
    models_dir: Path = Field(default=Path("trained_models"))
    reports_dir: Path = Field(default=Path("artifacts/reports"))

    @validator("base_dir", "models_dir", "reports_dir", pre=True)
    def _resolve(cls, value: Path | str) -> Path:  # type: ignore[misc]
        return _resolve_path(value)


class AppConfig(BaseModel):
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    svm: SVMConfig = SVMConfig()
    cnn: CNNConfig = CNNConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """
    Load configuration from a YAML file, falling back to defaults if missing.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig(**data)
    return AppConfig()


def dump_config(config: AppConfig, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    """Persist the current configuration to a YAML file."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.dict(), f, sort_keys=False)

