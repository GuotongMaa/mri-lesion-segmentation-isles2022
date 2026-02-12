#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python3 -m mri_seg.train \
  --data-dir data/train_subset_flat \
  --epochs 4 \
  --batch-size 1 \
  --patch-size 48 48 48 \
  --patches-per-volume 1 \
  --learning-rate 1e-3 \
  --positive-patch-ratio 0.8 \
  --num-workers 0
