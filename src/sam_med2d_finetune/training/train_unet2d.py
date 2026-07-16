"""Fair conventional 2D U-Net training for BraTS baseline comparison.

Follows the frozen protocol in the baselines PRD:
- Same paper_v1 split, cache, normalization, and augmentation as A0-A3
- BCE + SoftDice loss, equal weight mean over ET/TC/WT
- AdamW, lr=1e-3, wd=1e-5, ReduceLROnPlateau
- Max 100 epochs, early stopping 20
- Checkpoint selection by raw patient-level macro-Dice on validation
"""

import argparse
import csv
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam_med2d_finetune.models.unet2d import UNet2D
from sam_med2d_finetune.training.metrics import SegMetrics
from sam_med2d_finetune.training.multitask_dataset import BraTSDataset
from sam_med2d_finetune.utils.cli import str_to_bool
from sam_med2d_finetune.utils.training import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D U-Net baseline for BraTS segmentation.")
    parser.add_argument("--train_data_path", required=True, help="BraTS training data path.")
    parser.add_argument("--val_data_path", required=True, help="BraTS validation data path.")
    parser.add_argument("--work_dir", default="workdir_unet", help="Work directory.")
    parser.add_argument("--run_name", default="unet2d", help="Run name.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", type=str_to_bool, default=True)
    parser.add_argument("--disable_cudnn", type=str_to_bool, default=False)
    parser.add_argument("--cudnn_benchmark", type=str_to_bool, default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", type=str_to_bool, default=False)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--non_blocking_transfer", type=str_to_bool, default=False)
    parser.add_argument("--cache_root", type=str, default=None)
    parser.add_argument("--cache_max_cases", type=int, default=0)
    parser.add_argument("--negative_to_positive_ratio", type=float, default=0.3333333333333333)
    parser.add_argument("--dataset_seed", type=int, default=11519)
    parser.add_argument("--seed", type=int, default=11519)
    parser.add_argument("--deterministic", type=str_to_bool, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--grad_clip_norm", type=float, default=12.0)
    parser.add_argument("--smooth", type=float, default=1.0, help="Soft Dice smoothing constant.")
    parser.add_argument("--profile_gpu_utilization", type=str_to_bool, default=False)
    parser.add_argument("--profile_sample_interval", type=float, default=1.0)
    parser.add_argument("--save_epochs", type=int, nargs="*", default=[])
    parser.add_argument("--max_train_steps", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed, deterministic):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class UNetLoss(nn.Module):
    """BCE + SoftDice loss, equal weight mean over ET/TC/WT."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """logits: (B, 3, H, W), targets: (B, 3, H, W)"""
        loss_total = 0.0
        for c in range(logits.shape[1]):
            logit_c = logits[:, c:c + 1, :, :]
            target_c = targets[:, c:c + 1, :, :]
            bce_loss = self.bce(logit_c, target_c)
            probs = torch.sigmoid(logit_c)
            intersection = torch.sum(probs * target_c)
            union = torch.sum(probs) + torch.sum(target_c)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - dice
            loss_total += 0.5 * bce_loss + 0.5 * dice_loss
        return loss_total / logits.shape[1]


def to_device(batch_input, device, non_blocking=False):
    return {
        "image": batch_input["image"].float().to(device, non_blocking=non_blocking),
        "label": batch_input["label"].float().to(device, non_blocking=non_blocking),
    }


def train_one_epoch(model, optimizer, train_loader, criterion, args, epoch, scaler):
    model.train()
    total_loss = 0.0
    total_dice = np.zeros(3)
    step_count = 0
    start_time = time.perf_counter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
    for batched_input in pbar:
        batched_input = to_device(batched_input, args.device, args.non_blocking_transfer)
        images, labels = batched_input["image"], batched_input["label"]

        optimizer.zero_grad()
        with autocast(device_type=args.device.split(':')[0], enabled=args.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        loss_item = loss.item()
        total_loss += loss_item
        pbar.set_postfix(loss=f'{loss_item:.4f}')

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            binary = (probs > 0.5).float()
            for c in range(3):
                dice_c, _ = SegMetrics(binary[:, c:c + 1, :, :], labels[:, c:c + 1, :, :], ['dice', 'iou'])
                total_dice[c] += dice_c

        step_count += 1
        if args.max_train_steps is not None and step_count >= args.max_train_steps:
            break

    elapsed = time.perf_counter() - start_time
    avg_loss = total_loss / step_count
    avg_dice = total_dice / step_count
    return avg_loss, avg_dice, {"steps": step_count, "elapsed_seconds": elapsed}


@torch.no_grad()
def validate_one_epoch(model, val_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    total_dice = np.zeros(3)
    step_count = 0

    for batched_input in tqdm(val_loader, desc="Validating"):
        batched_input = to_device(batched_input, args.device)
        images, labels = batched_input["image"], batched_input["label"]

        with autocast(device_type=args.device.split(':')[0], enabled=args.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        binary = (probs > 0.5).float()
        for c in range(3):
            dice_c, _ = SegMetrics(binary[:, c:c + 1, :, :], labels[:, c:c + 1, :, :], ['dice', 'iou'])
            total_dice[c] += dice_c
        step_count += 1

    avg_loss = total_loss / step_count
    avg_dice = total_dice / step_count
    return avg_loss, avg_dice


def main(args):
    log_dir = os.path.join(args.work_dir, "logs", args.run_name)
    model_dir = os.path.join(args.work_dir, "models", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger = get_logger(os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")

    seed_everything(args.seed, args.deterministic)
    torch.backends.cudnn.enabled = not args.disable_cudnn
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.deterministic

    model = UNet2D(in_channels=4, num_classes=3).to(args.device)
    param_counts = model.count_parameters()
    logger.info(f"UNet2D parameters: total={param_counts['total']}, trainable={param_counts['trainable']}")
    print(f"UNet2D: {param_counts['total']:,} total params, {param_counts['trainable']:,} trainable")

    train_dataset = BraTSDataset(
        data_path=args.train_data_path, image_size=args.image_size,
        num_classes=3, mode='train', cache_root=args.cache_root,
        cache_max_cases=args.cache_max_cases,
        negative_to_positive_ratio=args.negative_to_positive_ratio,
        negative_prompt_box='zero', sample_seed=args.dataset_seed,
    )
    val_dataset = BraTSDataset(
        data_path=args.val_data_path, image_size=args.image_size,
        num_classes=3, mode='val', cache_root=args.cache_root,
        cache_max_cases=args.cache_max_cases,
        negative_to_positive_ratio=0.0, negative_prompt_box='zero',
        sample_seed=args.dataset_seed,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=train_generator,
        persistent_workers=args.persistent_workers,
        **(dict(prefetch_factor=args.prefetch_factor) if args.prefetch_factor is not None else {}),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=val_generator,
        persistent_workers=args.persistent_workers,
        **(dict(prefetch_factor=args.prefetch_factor) if args.prefetch_factor is not None else {}),
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.scheduler_factor,
        patience=args.scheduler_patience, verbose=True,
    )
    criterion = UNetLoss(smooth=args.smooth)
    scaler = GradScaler(enabled=args.use_amp and 'cuda' in args.device)

    csv_path = os.path.join(log_dir, "metrics.csv")
    csv_header = [
        'epoch', 'train_loss', 'val_loss', 'train_dice_ET', 'train_dice_TC', 'train_dice_WT',
        'val_dice_ET', 'val_dice_TC', 'val_dice_WT', 'val_dice_mean',
        'lr', 'train_elapsed_seconds',
    ]

    best_val_dice = -1.0
    best_epoch = -1
    early_stop_counter = 0

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    for epoch in range(args.epochs):
        train_loss, train_dice, runtime = train_one_epoch(
            model, optimizer, train_loader, criterion, args, epoch, scaler,
        )
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion, args)
        val_dice_mean = float(np.mean(val_dice))

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_dice_mean)

        row = [
            epoch + 1, train_loss, val_loss,
            float(train_dice[0]), float(train_dice[1]), float(train_dice[2]),
            float(val_dice[0]), float(val_dice[1]), float(val_dice[2]),
            val_dice_mean, current_lr, runtime['elapsed_seconds'],
        ]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        logger.info(
            f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Dice: ET={val_dice[0]:.4f} TC={val_dice[1]:.4f} WT={val_dice[2]:.4f} "
            f"Mean={val_dice_mean:.4f} | LR: {current_lr:.2e}"
        )

        # Checkpoint selection by val_dice_mean
        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
        else:
            early_stop_counter += 1

        # Save epoch snapshots
        if epoch + 1 in args.save_epochs:
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, f"epoch_{epoch + 1:03d}.pth"),
            )

        # Early stopping
        if early_stop_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {args.early_stopping_patience} epochs).")
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    logger.info(f"Best checkpoint: epoch {best_epoch}, val_dice_mean={best_val_dice:.4f}")
    print(f"Best checkpoint: epoch {best_epoch}, val_dice_mean={best_val_dice:.4f}")


if __name__ == "__main__":
    main(parse_args())
