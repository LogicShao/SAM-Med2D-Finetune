import argparse
import csv
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from peft import get_peft_model, LoraConfig
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader import SAMDataset, custom_collate_fn
from metrics import SegMetrics
from segment_anything import sam_model_registry
from utils import FocalDiceloss_IoULoss, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="统一的 SAM-Med2D 微调框架")
    parser.add_argument("--finetune_method", type=str, required=True, choices=['adapter', 'lora'])
    parser.add_argument("--data_path", type=str, default="data_brats_processed")
    parser.add_argument("--work_dir", type=str, default="workdir_brats")
    parser.add_argument("--run_name", type=str, default="sam_finetune")
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument('--lora_target_modules', nargs='+', default=['qkv'],
                        help='应用 LoRA 的目标层。对于 SAM ViT，通常是 "qkv"。')
    parser.add_argument("--encoder_adapter", type=bool, default=True)
    parser.add_argument("--use_amp", action='store_true', help="启用自动混合精度训练 (AMP)", default=True)
    args = parser.parse_args()
    args.run_name = f"{args.run_name}_{args.finetune_method}"
    return args


def train_one_epoch(model, optimizer, train_loader, criterion, args, epoch, scaler):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0
    # 将 tqdm 包装器移到外面，并只显示一个进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
    for batched_input in pbar:
        if batched_input is None: continue
        # --- 修复 RuntimeError: 将数据类型转换为 float32 ---
        images = batched_input["image"].to(args.device, dtype=torch.float32)
        labels = batched_input["label"].to(args.device)
        boxes = batched_input["boxes"].to(args.device)

        optimizer.zero_grad()

        with autocast(device_type=args.device.split(':')[0], enabled=args.use_amp):
            image_embeddings = model.image_encoder(images)
            sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
            masks, iou_pred = model.mask_decoder(image_embeddings, model.prompt_encoder.get_dense_pe(), sparse, dense,
                                                 False)
            upscaled_masks = F.interpolate(masks, (args.image_size, args.image_size), mode="bilinear",
                                           align_corners=False)
            loss = criterion(upscaled_masks, labels, iou_pred)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_item = loss.item()
        total_loss += loss_item
        with torch.no_grad():
            binary_masks = (torch.sigmoid(upscaled_masks) > 0.5).float()
            dice, iou = SegMetrics(binary_masks, labels, ['dice', 'iou'])
            total_dice += dice
            total_iou += iou

        # 在进度条上显示实时损失
        pbar.set_postfix(loss=f'{loss_item:.4f}')

    return total_loss / len(train_loader), total_dice / len(train_loader), total_iou / len(train_loader)


def validate_one_epoch(model, val_loader, criterion, args):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    with torch.no_grad():
        for batched_input in tqdm(val_loader, desc="Validating"):
            if batched_input is None: continue
            images = batched_input["image"].to(args.device, dtype=torch.float32)
            labels = batched_input["label"].to(args.device)
            boxes = batched_input["boxes"].to(args.device)

            with autocast(device_type=args.device.split(':')[0], enabled=args.use_amp):
                image_embeddings = model.image_encoder(images)
                sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
                masks, iou_pred = model.mask_decoder(image_embeddings, model.prompt_encoder.get_dense_pe(), sparse,
                                                     dense, False)
                upscaled_masks = F.interpolate(masks, (args.image_size, args.image_size), mode="bilinear",
                                               align_corners=False)
                loss = criterion(upscaled_masks, labels, iou_pred)

            total_loss += loss.item()
            binary_masks = (torch.sigmoid(upscaled_masks) > 0.5).float()
            dice, iou = SegMetrics(binary_masks, labels, ['dice', 'iou'])
            total_dice += dice
            total_iou += iou

    return total_loss / len(val_loader), total_dice / len(val_loader), total_iou / len(val_loader)


def plot_metrics(csv_path, save_dir):
    df = pd.read_csv(csv_path)
    epochs = df['epoch']
    metrics_to_plot = ['loss', 'dice', 'iou']
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, df[f'train_{metric}'], 'b-o', label=f'Train {metric.capitalize()}')
        plt.plot(epochs, df[f'val_{metric}'], 'r-o', label=f'Validation {metric.capitalize()}')
        plt.title(f'Training & Validation {metric.capitalize()}')
        plt.xlabel('Epoch');
        plt.ylabel(metric.capitalize());
        plt.legend();
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f'{metric}_curve.png'))
        plt.close()


def main(args):
    log_dir = os.path.join(args.work_dir, "logs", args.run_name)
    model_dir = os.path.join(args.work_dir, "models", args.run_name)
    plot_dir = os.path.join(args.work_dir, "plots", args.run_name)
    os.makedirs(log_dir, exist_ok=True);
    os.makedirs(model_dir, exist_ok=True);
    os.makedirs(plot_dir, exist_ok=True)

    logger = get_logger(os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))

    # --- 简化信息输出：这些信息将只写入日志文件 ---
    logger.info("--- Command Line Arguments ---")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info("----------------------------\n")

    is_amp_enabled = args.use_amp and ('cuda' in args.device)
    scaler = GradScaler(enabled=is_amp_enabled)
    logger.info(f"Automatic Mixed Precision (AMP) enabled: {is_amp_enabled}")

    model = sam_model_registry[args.model_type](args).to(args.device)

    if args.sam_checkpoint and os.path.isfile(args.sam_checkpoint):
        try:
            with open(args.sam_checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=args.device, weights_only=False)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                model.load_state_dict(state_dict, strict=False)
            logger.info(f"成功加载预训练权重: {args.sam_checkpoint}")
            print(f"成功加载预训练权重: {args.sam_checkpoint}")
        except Exception as e:
            logger.error("加载权重失败! 详细错误信息如下:")
            logger.error(e)
            print("加载权重失败，详情请查看日志文件。")
    else:
        logger.warning("未提供或找不到预训练权重。")
        print("警告: 未提供预训练权重。")

    if args.finetune_method == 'lora':
        logger.info("应用 LoRA 配置...")
        for n, p in model.named_parameters(): p.requires_grad = False

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=0.1,
            bias="none"
        )
        model.image_encoder = get_peft_model(model.image_encoder, lora_config)
    elif args.finetune_method == 'adapter':
        logger.info("使用 Adapter 微调...")
        for n, p in model.named_parameters():
            if 'Adapter' not in n and 'mask_decoder' not in n and 'prompt_encoder' not in n:
                p.requires_grad = False

    logger.info("\n--- 可训练参数列表 ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for n, p in model.named_parameters():
        if p.requires_grad: logger.info(f"  - {n}")
    logger.info(f"总可训练参数量: {total_params / 1e6:.2f}M\n")
    print(f"微调方法: {args.finetune_method} | 总可训练参数量: {total_params / 1e6:.2f}M")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    train_dataset = SAMDataset(args.data_path, image_size=args.image_size, mode='train', mask_num=1)
    val_dataset = SAMDataset(args.data_path, image_size=args.image_size, mode='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    csv_path = os.path.join(log_dir, "metrics.csv")
    csv_header = ['epoch', 'train_loss', 'val_loss', 'train_dice', 'val_dice', 'train_iou', 'val_iou']
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    best_val_metric = -1.0
    patience_counter = 0

    print(f"\n开始训练... 日志文件位于: {log_dir}")
    for epoch in range(args.epochs):
        train_loss, train_dice, train_iou = train_one_epoch(model, optimizer, train_loader, criterion, args, epoch,
                                                            scaler)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, args)

        # --- 简化信息输出：每轮只打印一行最重要的结果 ---
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        logger.info(
            f"Epoch {epoch + 1} | Train -> Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        logger.info(f"Epoch {epoch + 1} | Val   -> Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        row = [epoch + 1, train_loss, val_loss, train_dice, val_dice, train_iou, val_iou]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([f'{v:.4f}' if isinstance(v, float) else v for v in row])

        if val_dice > best_val_metric:
            best_val_metric = val_dice
            patience_counter = 0
            if args.finetune_method == 'lora':
                model.image_encoder.save_pretrained(os.path.join(model_dir, 'lora_encoder_best'))
            else:  # adapter
                torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            logger.info(f"新最佳模型已保存, Val Dice: {best_val_metric:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print("早停机制已触发。")
                logger.info("Early stopping triggered.")
                break

    print(f"\n训练结束。最佳验证 Dice: {best_val_metric:.4f}")
    logger.info(f"Training finished. Best validation Dice: {best_val_metric:.4f}")

    print("正在生成指标曲线图...")
    plot_metrics(csv_path, plot_dir)
    print(f"曲线图已保存到: {plot_dir}")
    logger.info(f"Plots saved to {plot_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
