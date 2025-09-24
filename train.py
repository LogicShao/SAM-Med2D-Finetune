import argparse
import csv
import datetime
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader import TrainingDataset, TestingDataset, stack_dict_batched
from metrics import SegMetrics
from segment_anything import sam_model_registry
from utils import FocalDiceloss_IoULoss, get_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")  # Adjusted for demo
    parser.add_argument("--batch_size", type=int, default=2, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number for training")
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="output multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--use_amp", type=bool, default=True, help="use amp")
    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif isinstance(value, list) or isinstance(value, torch.Size):
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.set_grad_enabled(not decoder_iter):
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask and iou_predictions is not None:
        # The iou_predictions might be of shape (B, num_masks=3)
        # We select the best mask based on the predicted IoU
        best_mask_indices = torch.argmax(iou_predictions, dim=1)
        # low_res_masks shape (B, num_masks=3, H, W) -> (B, 1, H, W)
        low_res_masks = torch.stack(
            [low_res_masks[i, best_mask_indices[i]] for i in range(len(best_mask_indices))]).unsqueeze(1)
        # Update iou_predictions to match the selected mask
        iou_predictions = torch.gather(iou_predictions, 1, best_mask_indices.unsqueeze(1))

    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)
    return masks, low_res_masks, iou_predictions


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler):
    model.train()
    train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)

    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        # Randomly choose between box and point prompts
        prompt_flag = "boxes" if random.random() > 0.5 else "point"
        if prompt_flag == "boxes":
            batched_input["point_coords"] = None
        else:
            batched_input["boxes"] = None

        # Freeze/unfreeze parts of the model
        for n, value in model.image_encoder.named_parameters():
            value.requires_grad = "Adapter" in n

        with autocast(device_type=args.device, dtype=torch.float16, enabled=args.use_amp):
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(batched_input["image"])
            image_embeddings_repeat = image_embeddings.repeat_interleave(args.mask_num, dim=0)
            masks, _, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings_repeat)
            loss = criterion(masks, labels, iou_predictions)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_losses.append(loss.item())
        train_loader.set_postfix(train_loss=loss.item())
        batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + batch_metrics[i] for i in range(len(args.metrics))]

    return np.mean(train_losses), [m / len(train_loader) for m in train_iter_metrics]


def validate_one_epoch(args, model, val_loader, criterion):
    model.eval()
    val_loader = tqdm(val_loader, desc="Validating")
    val_losses = []
    val_iter_metrics = [0] * len(args.metrics)

    with torch.no_grad():
        for batch, batched_input in enumerate(val_loader):
            batched_input = to_device(batched_input, args.device)
            labels = batched_input["label"]

            with autocast(device_type=args.device, dtype=torch.float16, enabled=args.use_amp):
                image_embeddings = model.image_encoder(batched_input["image"])
                masks, _, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                loss = criterion(masks, labels, iou_predictions)

            val_losses.append(loss.item())
            val_loader.set_postfix(val_loss=loss.item())
            batch_metrics = SegMetrics(masks, labels, args.metrics)
            val_iter_metrics = [val_iter_metrics[i] + batch_metrics[i] for i in range(len(args.metrics))]

    return np.mean(val_losses), [m / len(val_loader) for m in val_iter_metrics]


def plot_metrics(csv_path, save_dir, metrics):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}. Skipping plotting.")
        return

    epochs = df['epoch']
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_loss'], marker='o', label='Train Loss')
    plt.plot(epochs, df['val_loss'], marker='o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df[f'train_{metric}'], marker='o', label=f'Train {metric.capitalize()}')
        plt.plot(epochs, df[f'val_{metric}'], marker='o', label=f'Validation {metric.capitalize()}')
        plt.title(f'Training and Validation {metric.capitalize()}')
        plt.xlabel('Epoch');
        plt.ylabel(metric.capitalize());
        plt.legend();
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{metric}_curve.png'))
        plt.close()


def main(args):
    model = sam_model_registry[args.model_type](args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()
    # 修正 GradScaler 初始化
    scaler = GradScaler(device=args.device, enabled=args.use_amp)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)

    if args.resume is not None:
        try:
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model'])
            print(f"******* Loaded model checkpoint from {args.resume}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # --- 数据加载 ---
    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1,
                                    mask_num=args.mask_num, requires_name=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = TestingDataset(args.data_path, image_size=args.image_size, mode='test', requires_name=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f'******* Train data: {len(train_dataset)}, Validation data: {len(val_dataset)}')

    # --- 目录设置 ---
    log_dir = os.path.join(args.work_dir, "logs", args.run_name)
    model_dir = os.path.join(args.work_dir, "models", args.run_name)
    plot_dir = os.path.join(args.work_dir, "plots", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    loggers = get_logger(os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))
    csv_path = os.path.join(log_dir, "metrics.csv")
    csv_header = ['epoch', 'train_loss', 'val_loss'] + [f'train_{m}' for m in args.metrics] + [f'val_{m}' for m in
                                                                                               args.metrics]

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    best_val_loss = float('inf')
    best_model_path = os.path.join(model_dir, "best_model.pth")

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_metrics_values = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion,
                                                           scaler)
        val_loss, val_metrics_values = validate_one_epoch(args, model, val_loader, criterion)

        if args.lr_scheduler:
            scheduler.step()

        log_message = (
                f"Epoch: {epoch + 1}, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Train Metrics: " + ", ".join(
            [f"{m}: {v:.4f}" for m, v in zip(args.metrics, train_metrics_values)]) + ", "
                                                                                     f"Val Metrics: " + ", ".join(
            [f"{m}: {v:.4f}" for m, v in zip(args.metrics, val_metrics_values)])
        )
        loggers.info(log_message)

        row_data = [epoch + 1, f'{train_loss:.4f}', f'{val_loss:.4f}'] + [f'{v:.4f}' for v in train_metrics_values] + [
            f'{v:.4f}' for v in val_metrics_values]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, best_model_path)
            loggers.info(
                f"****** New best model saved to {best_model_path} at epoch {epoch + 1} with val_loss {val_loss:.4f} ******")

        loggers.info(f"Epoch {epoch + 1} finished in {time.time() - start_time:.2f}s.\n")

    loggers.info("Training finished. Generating plots...")
    plot_metrics(csv_path, plot_dir, args.metrics)
    loggers.info(f"Plots saved to {plot_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
