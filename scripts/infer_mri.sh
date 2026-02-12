#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python3 -m mri_seg.infer \
  --checkpoint models/best_model.pt \
  --input-dir data/infer \
  --output-dir results/predictions \
  --threshold 0.5
