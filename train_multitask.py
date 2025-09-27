import argparse
import csv
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import SegMetrics
from multitask_dataset import BraTSDataset
from segment_anything import sam_model_registry
from utils import FocalDiceloss_IoULoss, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="多任务学习框架下的 SAM-Med2D 微调")

    # --- 核心选择参数 ---
    parser.add_argument("--finetune_method", type=str, required=True, choices=['adapter', 'lora'],
                        help="选择微调方法: 'adapter' 或 'lora'")

    # --- 路径和模型配置 ---
    parser.add_argument("--train_data_path", type=str, required=True, help="BraTS 训练集路径")
    parser.add_argument("--val_data_path", type=str, required=True, help="BraTS 验证集路径")
    parser.add_argument("--work_dir", type=str, default="workdir_brats", help="工作目录")
    parser.add_argument("--run_name", type=str, default="sam_multitask", help="运行名称，会自动加上方法后缀")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM 模型类型")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth",
                        help="预训练 SAM-Med2D 权重路径")

    # --- 训练超参数 ---
    parser.add_argument("--epochs", type=int, default=200, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument('--device', type=str, default='cuda', help="设备")
    parser.add_argument("--use_amp", action='store_true', default=True, help="启用自动混合精度训练 (AMP)")

    # --- 数据集和调试 ---
    parser.add_argument("--train_subset_size", type=int, default=None, help="使用训练集的前N个样本")
    parser.add_argument("--val_subset_size", type=int, default=None, help="使用验证集的前N个样本")

    # --- 早停机制 ---
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="验证集性能连续N轮不提升则早停")

    # --- LoRA/Adapter 参数 ---
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument('--lora_target_modules', nargs='+', default=['qkv'], help='应用 LoRA 的目标层')
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="模型是否包含 Adapter (用于构建和冻结)")

    # --- BraTS 特定的参数定义 ---
    parser.add_argument("--num_classes", type=int, default=3, help="BraTS 的分割类别数 (ET, TC, WT)")
    parser.add_argument("--input_channels", type=int, default=4, help="BraTS 输入模态数")

    args = parser.parse_args()
    args.run_name = f"{args.run_name}_{args.finetune_method}"
    return args


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key in ['image', 'label']:
                device_input[key] = value.float().to(device)
            elif isinstance(value, (list, torch.Size)):
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def train_one_epoch(model, optimizer, train_loader, criterion, args, epoch, scaler):
    model.train()
    total_loss = 0.0
    total_dice = np.zeros(args.num_classes)
    total_iou = np.zeros(args.num_classes)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
    for batched_input in pbar:
        batched_input = to_device(batched_input, args.device)
        images, labels = batched_input["image"], batched_input["label"]

        optimizer.zero_grad()

        with autocast(device_type=args.device.split(':')[0], enabled=args.use_amp):
            image_embeddings = model.image_encoder(images)

            accumulated_loss = 0
            all_class_masks = []
            for c in range(args.num_classes):
                boxes_c = batched_input["boxes"][:, c, :]
                labels_c = labels[:, c:c + 1, :, :]

                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None, boxes=boxes_c.unsqueeze(1), masks=None)

                # --- 核心修正 2：移除 .detach() ---
                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,  # 直接传递，不分离计算图
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False)

                upscaled_masks = F.interpolate(
                    low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)

                loss = criterion(upscaled_masks, labels_c, iou_predictions)
                accumulated_loss += loss
                all_class_masks.append(upscaled_masks)

            final_loss = accumulated_loss / args.num_classes

        scaler.scale(final_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_item = final_loss.item()
        total_loss += loss_item
        pbar.set_postfix(loss=f'{loss_item:.4f}')

        with torch.no_grad():
            for c in range(args.num_classes):
                binary_masks = (torch.sigmoid(all_class_masks[c]) > 0.5).float()
                dice_c, iou_c = SegMetrics(binary_masks, labels[:, c:c + 1, :, :], ['dice', 'iou'])
                total_dice[c] += dice_c
                total_iou[c] += iou_c

    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    return avg_loss, avg_dice, avg_iou


def validate_one_epoch(model, val_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    total_dice = np.zeros(args.num_classes)
    total_iou = np.zeros(args.num_classes)

    with torch.no_grad():
        for batched_input in tqdm(val_loader, desc="Validating"):
            batched_input = to_device(batched_input, args.device)
            images, labels = batched_input["image"], batched_input["label"]

            with autocast(device_type=args.device.split(':')[0], enabled=args.use_amp):
                image_embeddings = model.image_encoder(images)
                accumulated_loss = 0
                for c in range(args.num_classes):
                    boxes_c = batched_input["boxes"][:, c, :]
                    labels_c = labels[:, c:c + 1, :, :]

                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None, boxes=boxes_c.unsqueeze(1), masks=None)

                    low_res_masks, iou_predictions = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False)

                    upscaled_masks = F.interpolate(
                        low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)

                    loss = criterion(upscaled_masks, labels_c, iou_predictions)
                    accumulated_loss += loss

                    binary_masks = (torch.sigmoid(upscaled_masks) > 0.5).float()
                    dice_c, iou_c = SegMetrics(binary_masks, labels_c, ['dice', 'iou'])
                    total_dice[c] += dice_c
                    total_iou[c] += iou_c

                total_loss += (accumulated_loss / args.num_classes).item()

    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    return avg_loss, avg_dice, avg_iou


def plot_metrics(csv_path, save_dir):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}. Skipping plotting.")
        return

    epochs = df['epoch']

    # 绘制 Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs, df['val_loss'], 'r-o', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 绘制 Mean Dice
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_dice_mean'], 'b-o', label='Train Mean Dice')
    plt.plot(epochs, df['val_dice_mean'], 'r-o', label='Validation Mean Dice')
    plt.title('Training & Validation Mean Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend();
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'dice_curve.png'))
    plt.close()

    # 绘制 Mean IoU
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, df['train_iou_mean'], 'b-o', label='Train Mean IoU')
    plt.plot(epochs, df['val_iou_mean'], 'r-o', label='Validation Mean IoU')
    plt.title('Training & Validation Mean IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'iou_curve.png'))
    plt.close()


def main(args):
    log_dir = os.path.join(args.work_dir, "logs", args.run_name)
    model_dir = os.path.join(args.work_dir, "models", args.run_name)
    plot_dir = os.path.join(args.work_dir, "plots", args.run_name)
    os.makedirs(log_dir, exist_ok=True);
    os.makedirs(model_dir, exist_ok=True);
    os.makedirs(plot_dir, exist_ok=True)

    logger = get_logger(os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))

    logger.info("--- Command Line Arguments ---")
    for key, value in vars(args).items(): logger.info(f"{key}: {value}")
    logger.info("----------------------------\n")

    is_amp_enabled = args.use_amp and ('cuda' in args.device)
    scaler = GradScaler(enabled=is_amp_enabled)
    logger.info(f"Automatic Mixed Precision (AMP) enabled: {is_amp_enabled}")

    # 1. 首先，创建原始的模型架构 (输入通道为3)
    model = sam_model_registry[args.model_type](args).to(args.device)

    # 2. 其次，加载预训练权重。
    #    因为我们接下来会替换掉输入层，所以这里的加载错误可以被忽略。
    #    我们使用 strict=False 来避免因 patch_embed 不匹配而导致的崩溃。
    if args.sam_checkpoint and os.path.isfile(args.sam_checkpoint):
        try:
            with open(args.sam_checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=args.device, weights_only=False)
                if 'model' in state_dict: state_dict = state_dict['model']

                # 使用 strict=False，它会加载所有形状匹配的层，并忽略不匹配的层
                model.load_state_dict(state_dict, strict=False)

            print(f"成功加载预训练权重 (忽略了输入层): {args.sam_checkpoint}")
            logger.info(f"成功加载预训练权重 (忽略了输入层): {args.sam_checkpoint}")
        except Exception as e:
            print(f"加载权重失败，详情请查看日志: {e}")
            logger.error(f"加载权重失败: {e}")
            # 这里可以选择是否在加载失败时退出
            return

    # 3. 最后，修改模型输入层以适应4通道 BraTS 数据
    #    这一步会替换掉 patch_embed.proj，其权重是新初始化的。
    with torch.no_grad():
        original_proj = model.image_encoder.patch_embed.proj
        # 创建一个新的4通道输入卷积层
        new_proj = nn.Conv2d(
            in_channels=args.input_channels,
            out_channels=original_proj.out_channels,
            kernel_size=original_proj.kernel_size,
            stride=original_proj.stride,
            padding=original_proj.padding,
            bias=(original_proj.bias is not None)
        ).to(args.device)

        # 初始化新层的权重：取原3通道权重的平均值，并复制到4个通道上
        # 这是一种常见的、比随机初始化更好的策略
        avg_weights = original_proj.weight.clone().detach().mean(dim=1, keepdim=True)
        new_proj.weight.copy_(avg_weights.repeat(1, args.input_channels, 1, 1))
        if original_proj.bias is not None:
            new_proj.bias.copy_(original_proj.bias)

        # 替换模型中的旧层
        model.image_encoder.patch_embed.proj = new_proj

    logger.info("模型输入层已成功修改为接收4通道输入。")
    print("模型输入层已成功修改为接收4通道输入。")

    # --- 根据方法选择微调策略 ---
    if args.finetune_method == 'lora':
        logger.info("应用 LoRA 配置...")
        for n, p in model.named_parameters(): p.requires_grad = False
        lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules)
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

    # --- 使用 multitask_dataset ---
    train_dataset = BraTSDataset(data_path=args.train_data_path, image_size=args.image_size,
                                 num_classes=args.num_classes, mode='train', subset_size=args.train_subset_size)
    val_dataset = BraTSDataset(data_path=args.val_data_path, image_size=args.image_size, num_classes=args.num_classes,
                               mode='val', subset_size=args.val_subset_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # --- 统一的日志和 CSV 记录 ---
    csv_path = os.path.join(log_dir, "metrics.csv")
    csv_header = ['epoch', 'train_loss', 'val_loss', 'train_dice_mean', 'val_dice_mean', 'train_iou_mean',
                  'val_iou_mean',
                  'val_dice_ET', 'val_dice_TC', 'val_dice_WT', 'val_iou_ET', 'val_iou_TC', 'val_iou_WT']
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    best_val_dice = -1.0
    patience_counter = 0

    print(f"\n开始训练... 日志文件位于: {log_dir}")
    for epoch in range(args.epochs):
        train_loss, train_dice, train_iou = train_one_epoch(model, optimizer, train_loader, criterion, args, epoch,
                                                            scaler)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, args)

        train_dice_mean, train_iou_mean = train_dice.mean(), train_iou.mean()
        val_dice_mean, val_iou_mean = val_dice.mean(), val_iou.mean()

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | Val Dice: {val_dice_mean:.4f} | ET: {val_dice[0]:.4f}, TC: {val_dice[1]:.4f}, WT: {val_dice[2]:.4f}")
        logger.info(f"Epoch {epoch + 1} | Train -> Loss: {train_loss:.4f}, Mean Dice: {train_dice_mean:.4f}, Mean IoU: {train_iou_mean:.4f}")
        logger.info(f"Epoch {epoch + 1} | Val   -> Loss: {val_loss:.4f}, Mean Dice: {val_dice_mean:.4f}, Mean IoU: {val_iou_mean:.4f}")
        logger.info(f"Val Dice per class -> ET: {val_dice[0]:.4f}, TC: {val_dice[1]:.4f}, WT: {val_dice[2]:.4f}")
        logger.info(f"Val IoU per class  -> ET: {val_iou[0]:.4f}, TC: {val_iou[1]:.4f}, WT: {val_iou[2]:.4f}\n")

        row_data = [epoch + 1, train_loss, val_loss, train_dice_mean, val_dice_mean, train_iou_mean, val_iou_mean,
                    val_dice[0], val_dice[1], val_dice[2], val_iou[0], val_iou[1], val_iou[2]]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([f'{v:.4f}' if isinstance(v, (float, np.floating)) else v for v in row_data])

        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            patience_counter = 0
            # --- 核心修正：根据微调方法选择正确的保存方式 ---
            if args.finetune_method == 'lora':
                # LoRA 只保存被 PEFT 包装过的 adapter 部分
                # 注意：我们包装的是 model.image_encoder
                save_dir = os.path.join(model_dir, 'lora_adapters')
                model.image_encoder.save_pretrained(save_dir)
                logger.info(f"新最佳 LoRA adapters 已保存到 {save_dir} (Val Dice: {best_val_dice:.4f})")
            else:  # finetune_method == 'adapter'
                # Adapter 参数是模型 state_dict 的一部分，所以保存整个 state_dict
                save_path = os.path.join(model_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                logger.info(f"新最佳模型 (Adapter) 已保存到 {save_path} (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print("早停机制已触发。")
                logger.info("Early stopping triggered.")
                break

    print(f"\n训练结束。最佳验证 Mean Dice: {best_val_dice:.4f}")
    logger.info(f"Training finished. Best validation Mean Dice: {best_val_dice:.4f}")

    print("正在生成指标曲线图...")
    plot_metrics(csv_path, plot_dir)
    print(f"曲线图已保存到: {plot_dir}")
    logger.info(f"Plots saved to {plot_dir}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
