from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("results")
    checkpoint_dir: Path = Path("models")
    image_suffix: str = "_T1w.nii.gz"
    mask_suffix: str = "_label-L_desc-T1lesion_mask.nii.gz"

    batch_size: int = 2
    patch_size: tuple[int, int, int] = (96, 96, 96)
    patches_per_volume: int = 8
    num_workers: int = 0

    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    positive_patch_ratio: float = 0.7
    grad_clip_norm: float = 1.0

    base_channels: int = 16
    deep_supervision: bool = False

    train_split: float = 0.8
    random_seed: int = 42

    threshold: float = 0.5
    use_amp: bool = True


@dataclass
class InferConfig:
    checkpoint_path: Path = Path("models/best_model.pt")
    input_dir: Path = Path("data/infer")
    output_dir: Path = Path("results/predictions")
    image_suffix: str = "_T1w.nii.gz"
    patch_size: tuple[int, int, int] = (96, 96, 96)
    stride: tuple[int, int, int] = (48, 48, 48)
    threshold: float = 0.5
    base_channels: int = 16
