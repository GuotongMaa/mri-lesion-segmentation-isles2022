#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python3 -m mri_seg.train \
  --data-dir data/train \
  --epochs 60 \
  --batch-size 2 \
  --patch-size 96 96 96 \
  --patches-per-volume 8 \
  --learning-rate 1e-3 \
  --positive-patch-ratio 0.7
