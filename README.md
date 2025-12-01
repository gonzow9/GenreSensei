# GTZAN Genre Classifier

GTZAN genre-classification with pipelines, interpretability hooks, and docs.

## Why I made this
- I grew up playing piano from age 8, grinding through exams and Eisteddfods; this project asks whether machines can recognise and reason about the music I love.
- Baselines (MFCC + SVM) vs. deep (CNN) vs. extensible embeddings (OpenL3/CLAP).
- Designed for fast verification: cached MFCCs, reproducible seeds, and artifact logging.

## Quickstart
```
python3.11 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# prepare cache (MFCC extraction) and train everything
python -m genre_classifier train --target both --prepare

# predict a new file
python -m genre_classifier predict liebestraum.mp3
```

## Commands
- `prepare`: extract MFCCs to `data_gtzan.json`
- `train --target [svm|cnn|both]`: train models, save to `trained_models/`, metrics + plots to `artifacts/reports/`
- `predict <audio>`: SVM ensemble prediction, optional `--save <path>` for JSON summary
- `validate`: sanity-check dataset layout
- Console script: once installed, `genre-classify train --target both --prepare`
- Could try UI: `streamlit run scripts/streamlit_app.py` (install `streamlit` separately)

## Project layout
- `genre_classifier/` – package modules (`data`, `models`, `train`, `predict`, `interpretability`, `reporting`, `config`, `cli`)
- `config.yaml` – central configuration (paths, hyperparams)
- `artifacts/` – metrics, plots, saliency maps, reports (git-friendly)
- `trained_models/` – saved SVM and CNN weights
- `experiments/` – notebooks for baselines + error analysis
- `tests/` – small sanity tests (`pytest -q`)
- `scripts/reproduce.sh` – tiny end-to-end smoke run on a small subset

## Reproducibility
- One source of truth config (`config.yaml`) parsed by Pydantic.
- `Makefile` targets: `prepare`, `train-svm`, `train-cnn`, `predict`, `test`.
- All random seeds set via `config.training.random_state`.
- Metrics persisted to JSON; confusion matrices + saliency plots saved to `artifacts/reports/`.

## Interpretability & evaluation
- SVM: ANOVA-based MFCC statistical feature ranking (`artifacts/reports/svm_stat_feature_importance.json`).
- CNN: gradient saliency over mel-spectrograms (`artifacts/reports/cnn_saliency.png`).
- Evaluation: per-genre precision/recall/F1 + confusion matrices for each model.

## Dataset stuff
- `validate` checks GTZAN folder structure and counts per genre.
- `max_files_per_genre` in `config.yaml` enables quick subset runs for reproducibility.

## Future work (scaffolding there)
- Swap CNN features for OpenL3/CLAP embeddings + linear head.
- Few-shot transfer to a second genre set; UMAP/t-SNE visualisations of embeddings.
- Lightweight Streamlit/Gradio front-end (see `scripts/reproduce.sh` for CLI stub).

## Extras
- Audio assets are large; keep artifacts/plots under version control, not raw audio.
- On Apple Silicon, TensorFlow ships wheels only up to Python 3.11; use `python3.11` for installs.
- If training fails due to GPU/CPU constraints, lower `cnn.epochs` and increase `max_files_per_genre` slicing.
