from __future__ import annotations

import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore_norm(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    mean = volume.mean()
    std = volume.std() + 1e-6
    return (volume - mean) / std


def load_nifti(path: Path) -> np.ndarray:
    arr = nib.load(str(path)).get_fdata(dtype=np.float32)
    return arr


def collect_pairs(data_dir: Path, image_suffix: str, mask_suffix: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(data_dir.rglob(f"*{image_suffix}")):
        mask_path = Path(str(image_path).replace(image_suffix, mask_suffix))
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    if not pairs:
        raise FileNotFoundError(
            f"No MRI image/mask pairs found in {data_dir}. "
            f"Expected file suffixes: image={image_suffix}, mask={mask_suffix}."
        )
    return pairs


def random_split_pairs(
    pairs: list[tuple[Path, Path]], train_split: float, random_seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    idx = list(range(len(pairs)))
    rng = random.Random(random_seed)
    rng.shuffle(idx)
    split_at = max(1, int(len(idx) * train_split))
    train_pairs = [pairs[i] for i in idx[:split_at]]
    val_pairs = [pairs[i] for i in idx[split_at:]]
    if not val_pairs:
        val_pairs = train_pairs[-1:]
        train_pairs = train_pairs[:-1]
    return train_pairs, val_pairs


class MRIPatchDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        patch_size: tuple[int, int, int],
        patches_per_volume: int,
        positive_patch_ratio: float,
        augment: bool,
    ) -> None:
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.positive_patch_ratio = positive_patch_ratio
        self.augment = augment

        self._cache: list[tuple[np.ndarray, np.ndarray]] = []
        for image_path, mask_path in pairs:
            image = zscore_norm(load_nifti(image_path))[None, ...]
            mask = (load_nifti(mask_path) > 0).astype(np.float32)[None, ...]
            self._cache.append((image, mask))

        self._index: list[int] = []
        for i in range(len(self._cache)):
            for _ in range(patches_per_volume):
                self._index.append(i)

    def __len__(self) -> int:
        return len(self._index)

    def _crop(
        self, x: np.ndarray, y: np.ndarray, force_positive: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        _, h, w, d = x.shape
        ph, pw, pd = self.patch_size

        if ph > h or pw > w or pd > d:
            raise ValueError(
                f"Patch size {self.patch_size} is larger than volume shape {(h, w, d)}"
            )

        for _ in range(20):
            z = random.randint(0, d - pd)
            y0 = random.randint(0, w - pw)
            x0 = random.randint(0, h - ph)

            xp = x[:, x0 : x0 + ph, y0 : y0 + pw, z : z + pd]
            yp = y[:, x0 : x0 + ph, y0 : y0 + pw, z : z + pd]

            if not force_positive or yp.sum() > 0:
                return xp, yp

        return xp, yp

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=3).copy()
            y = np.flip(y, axis=3).copy()
        return x, y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vol_idx = self._index[idx]
        x, y = self._cache[vol_idx]

        force_positive = random.random() < self.positive_patch_ratio
        xp, yp = self._crop(x, y, force_positive=force_positive)

        if self.augment:
            xp, yp = self._augment(xp, yp)

        return torch.from_numpy(xp), torch.from_numpy(yp)
