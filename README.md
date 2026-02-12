# MRI Lesion Segmentation (Flagship)

3D medical image segmentation project for lesion detection on ATLAS/ISLES-style MRI volumes. This project is now structured as a reproducible training pipeline rather than a coursework notebook.

## Problem
Segment small lesion regions in 3D MRI under severe foreground-background imbalance while keeping training feasible on constrained hardware.

## Technical Strategy
- Residual 3D U-Net (`src/mri_seg/model.py`) for stronger feature propagation.
- Lesion-aware patch sampling (`src/mri_seg/data.py`) to reduce empty-patch dominance.
- Combined Dice+BCE or Tversky loss (`src/mri_seg/losses.py`) for imbalance handling.
- Metrics logged per epoch: Dice, IoU, lesion recall (`src/mri_seg/metrics.py`).
- Early stopping + best checkpoint saving (`src/mri_seg/train.py`).

## Project Structure
- `notebooks/` experiment notebooks (`mri_deep_dive.ipynb` is the primary showcase).
- `src/mri_seg/` training and inference code.
- `scripts/` one-command train/infer scripts.
- `models/` model checkpoints (`best_model.pt`).
- `results/` metrics CSV and exported predictions.
- `data/` local dataset folders (`train/` and `infer/`).

## Quick Start
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Prepare data:
   - Put train images/masks in `data/train/` with suffixes `_T1w.nii.gz` and `_label-L_desc-T1lesion_mask.nii.gz`.
3. Train:
   - `./scripts/train_mri.sh`
4. Inference:
   - `./scripts/infer_mri.sh`

## Current Status
- Baseline notebook exists: `notebooks/mri_isles2022.ipynb`.
- Upgraded pipeline is implemented and ready to run: `src/mri_seg/`.
- Initial real run completed on a 16-case training subset for quick validation.

## Initial Metrics (Real Subset Run)
- Run setup: 16 cases from ATLAS train derivatives, 4 epochs, patch size `48x48x48`, batch size `1`.
- Best validation Dice: `0.3750`
- Best validation IoU: `0.3750`
- Best lesion recall: `0.3750`
- Metric log: `results/metrics.csv`
- Run summary: `results/subset_run_summary.txt`

## Loss Ablation (Real Subset Run)
- Dice+BCE (best for Dice): `val_dice=0.3750`, `val_iou=0.3750`, `val_recall=0.3750`
- Tversky (higher recall profile): `val_dice=0.0864`, `val_iou=0.0476`, `val_recall=0.8454`
- Ablation table: `results/ablation_summary.csv`

## Results Showcase
Current artifacts:
- `assets/dice_curve_subset.png`
- `assets/loss_curve_subset.png`
- `assets/ablation_val_dice.png`
- `assets/ablation_val_recall.png`

![Dice Curve](assets/dice_curve_subset.png)
![Loss Curve](assets/loss_curve_subset.png)
![Ablation Dice](assets/ablation_val_dice.png)
![Ablation Recall](assets/ablation_val_recall.png)

Next artifacts to add from full-data runs:
- `assets/overlay_case_*.png` showing MRI/GT/Prediction.

## Suggested Target for Flagship Positioning
Use the same validation split and report all three:
- `Val Dice` (primary)
- `Val IoU`
- `Lesion Recall`

This keeps the project technically credible for medical segmentation interviews and avoids overclaiming from a single metric.
