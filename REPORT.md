# Project Report

## Motivation
- Lifelong musician: started piano at 8, sat graded exams, and played Eisteddfods; I want to see how well machines can recognize the genres I grew up hearing and performing.
- Research question: how far can classical MFCC + SVM baselines go versus lightweight CNNs, and what do they actually listen to?

## Dataset
- GTZAN (10 genres, 100 clips/genre, 30s each). Known issues: duplicate tracks, artist overlap, occasional clipping and mislabels.
- Local root: `Data 2/genres_original` (configurable via `config.yaml`).
- Mitigation: dataset validator and optional `max_files_per_genre` for reproducible subsets.

## Methods
- Features: MFCCs (13 coeffs, 2048 FFT, hop 512), segment-wise extraction (default 5 segments per 30s track).
- Models:
  - SVM (RBF): three feature views – statistical (mean/std/min/max/median/p25/p75), flattened (full temporal), hybrid (stats + downsampled frames).
  - CNN: 3x conv blocks over MFCC "images" with batch-norm + dropout; Adam with 1e-4 LR.
- Seeds: `config.training.random_state` controls all RNGs.

## Experiments
- Split: stratified train/test = 75/25.
- Metrics: accuracy + per-genre precision/recall/F1; confusion matrices saved to `artifacts/reports/`.
- Artifacts:
  - `svm_metrics.json`, `svm_stats_confusion.png`, `svm_stat_feature_importance.json`
  - `cnn_metrics.json`, `cnn_confusion.png`, `cnn_saliency.png`
- Results (full GTZAN, stratified 75/25):
  - SVM (stats): accuracy = 0.80. Strong on classical/pop/jazz; main confusions: rock→metal/country/disco, hiphop→reggae, country→blues/disco.
  - SVM (flat): accuracy = 0.53. Unstable high-dimensional baseline; rock suffers heavy confusion across genres.
  - SVM (hybrid): accuracy = 0.67. Improves over flat, still weaker than stats on pop/classical.
  - CNN: accuracy = 0.67 (validation accuracy peaked at 0.68, finished at 0.67). Saliency map highlights mid/low MFCC bands, suggesting rhythmic/harmonic cues. This is intentionally a lightweight baseline (MFCC input, no augmentation, small 3-block CNN), so it trades accuracy for simplicity/repro speed.

## Error analysis
- Notebooks: `experiments/baselines.ipynb` (training curves, confusion matrices), `experiments/error_analysis.ipynb` (where models disagree, t-SNE/UMAP hooks).
- Checklist:
  - Listen to top-5 false positives/negatives.
  - Inspect saliency map for whether drums/vocals dominate decisions.
  - Cross-check duplicates/artist leakage using validator script.

## Future work
- Embedder comparison: OpenL3/CLAP embeddings + linear classifier baseline.
- Few-shot transfer: adapt model to a small secondary genre set; evaluate catastrophic forgetting.
- Robustness: pitch/time-stretch augmentations and calibration checks.
- Streamlit/Gradio front-end showing spectrogram + top-3 predictions live.
