from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mri_seg.config import TrainConfig
from mri_seg.data import MRIPatchDataset, collect_pairs, random_split_pairs, set_seed
from mri_seg.losses import DiceBCELoss, TverskyLoss
from mri_seg.metrics import dice_score, iou_score, lesion_recall
from mri_seg.model import ResidualUNet3D


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 3D MRI lesion segmentation model")
    p.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    p.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    p.add_argument("--checkpoint-dir", type=Path, default=TrainConfig.checkpoint_dir)
    p.add_argument("--image-suffix", type=str, default=TrainConfig.image_suffix)
    p.add_argument("--mask-suffix", type=str, default=TrainConfig.mask_suffix)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--base-channels", type=int, default=TrainConfig.base_channels)
    p.add_argument("--patch-size", type=int, nargs=3, default=TrainConfig.patch_size)
    p.add_argument("--patches-per-volume", type=int, default=TrainConfig.patches_per_volume)
    p.add_argument("--positive-patch-ratio", type=float, default=TrainConfig.positive_patch_ratio)
    p.add_argument("--threshold", type=float, default=TrainConfig.threshold)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--seed", type=int, default=TrainConfig.random_seed)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--use-tversky", action="store_true", help="Use Tversky loss instead of Dice+BCE")
    return p.parse_args()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip_norm: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    recall_sum = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

        with torch.no_grad():
            d = dice_score(logits, y, threshold=threshold).mean().item()
            i = iou_score(logits, y, threshold=threshold).mean().item()
            r = lesion_recall(logits, y, threshold=threshold).mean().item()

        bs = x.shape[0]
        loss_sum += loss.item() * bs
        dice_sum += d * bs
        iou_sum += i * bs
        recall_sum += r * bs
        n += bs

    return {
        "loss": loss_sum / max(n, 1),
        "dice": dice_sum / max(n, 1),
        "iou": iou_sum / max(n, 1),
        "recall": recall_sum / max(n, 1),
    }


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.data_dir, image_suffix=args.image_suffix, mask_suffix=args.mask_suffix)
    train_pairs, val_pairs = random_split_pairs(pairs, train_split=0.8, random_seed=args.seed)

    train_ds = MRIPatchDataset(
        train_pairs,
        patch_size=tuple(args.patch_size),
        patches_per_volume=args.patches_per_volume,
        positive_patch_ratio=args.positive_patch_ratio,
        augment=True,
    )
    val_ds = MRIPatchDataset(
        val_pairs,
        patch_size=tuple(args.patch_size),
        patches_per_volume=max(2, args.patches_per_volume // 2),
        positive_patch_ratio=0.5,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(base_channels=args.base_channels).to(device)

    criterion: torch.nn.Module
    if args.use_tversky:
        criterion = TverskyLoss(alpha=0.7, beta=0.3)
    else:
        criterion = DiceBCELoss(pos_weight=10.0, bce_weight=0.4)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_dice = -1.0
    wait = 0

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_dice",
                "train_iou",
                "train_recall",
                "val_loss",
                "val_dice",
                "val_iou",
                "val_recall",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_stats = run_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                args.threshold,
                scaler,
                grad_clip_norm=1.0,
            )
            val_stats = run_epoch(
                model,
                val_loader,
                None,
                criterion,
                device,
                args.threshold,
                scaler=None,
                grad_clip_norm=1.0,
            )
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": f"{train_stats['loss']:.6f}",
                "train_dice": f"{train_stats['dice']:.6f}",
                "train_iou": f"{train_stats['iou']:.6f}",
                "train_recall": f"{train_stats['recall']:.6f}",
                "val_loss": f"{val_stats['loss']:.6f}",
                "val_dice": f"{val_stats['dice']:.6f}",
                "val_iou": f"{val_stats['iou']:.6f}",
                "val_recall": f"{val_stats['recall']:.6f}",
                "lr": f"{lr:.8f}",
            }
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} "
                f"train_dice={train_stats['dice']:.4f} val_dice={val_stats['dice']:.4f} "
                f"val_iou={val_stats['iou']:.4f} val_recall={val_stats['recall']:.4f}"
            )

            if val_stats["dice"] > best_dice:
                best_dice = val_stats["dice"]
                wait = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_val_dice": best_dice,
                        "config": vars(args),
                    },
                    args.checkpoint_dir / "best_model.pt",
                )
            else:
                wait += 1

            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}, best val dice={best_dice:.4f}")
                break

    print(f"Training complete. Best val Dice={best_dice:.4f}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
