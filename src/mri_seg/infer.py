from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mri_seg.data import load_nifti, zscore_norm
from mri_seg.model import ResidualUNet3D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for MRI lesion segmentation")
    p.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    p.add_argument("--input-dir", type=Path, default=Path("data/infer"))
    p.add_argument("--output-dir", type=Path, default=Path("results/predictions"))
    p.add_argument("--image-suffix", type=str, default="_T1w.nii.gz")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--base-channels", type=int, default=16)
    return p.parse_args()


def infer_single(model: torch.nn.Module, volume: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(volume[None, None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return prob


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(base_channels=args.base_channels).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    image_paths = sorted(args.input_dir.rglob(f"*{args.image_suffix}"))
    if not image_paths:
        raise FileNotFoundError(f"No input MRI found in {args.input_dir} with suffix {args.image_suffix}")

    for image_path in image_paths:
        volume = zscore_norm(load_nifti(image_path))
        prob = infer_single(model, volume, device)
        pred = (prob >= args.threshold).astype(np.uint8)

        out_name = image_path.name.replace(args.image_suffix, "_predmask.nii.gz")
        out_path = args.output_dir / out_name

        # Preserve spatial metadata from source image
        src_nii = nib.load(str(image_path))
        out_nii = nib.Nifti1Image(pred, affine=src_nii.affine, header=src_nii.header)
        nib.save(out_nii, str(out_path))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
