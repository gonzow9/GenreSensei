#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end run. quicker set dataset.max_files_per_genre in config.yaml (eg 10).

python -m genre_classifier prepare
python -m genre_classifier train --target svm
python -m genre_classifier predict liebestraum.mp3 --save artifacts/predictions/liebestraum.json
