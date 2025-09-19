import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from BraTSDataset import BraTSDataset
from metrics import SegMetrics
from segment_anything import sam_model_registry
from utils import FocalDiceloss_IoULoss, get_logger

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SAM-Med2D on BraTS 2021 with Validation and Early Stopping")

    # --- 路径和模型配置 ---
    parser.add_argument("--train_data_path", type=str, required=True, help="BraTS 2021 训练集路径")
    parser.add_argument("--val_data_path", type=str, required=True, help="BraTS 2021 验证集路径")
    parser.add_argument("--work_dir", type=str, default="workdir_brats", help="工作目录")
    parser.add_argument("--run_name", type=str, default="sam-med2d-brats-lora", help="运行名称")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM 模型类型")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth",
                        help="预训练 SAM-Med2D 权重路径")

    # --- 训练超参数 ---
    parser.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率 (建议从较低值开始)")
    parser.add_argument('--device', type=str, default='cuda', help="设备")

    # --- 数据集和调试 ---
    parser.add_argument("--train_subset_size", type=int, default=None, help="使用训练集的前N个样本")
    parser.add_argument("--val_subset_size", type=int, default=None, help="使用验证集的前N个样本")

    # --- 早停机制 ---
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="验证集性能连续N轮不提升则早停")

    # --- LoRA 参数 ---
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'v_proj'], help='应用 LoRA 的目标层')
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="加载模型时是否构建 Adapter 层")

    # --- BraTS 特定的参数定义 ---
    parser.add_argument("--num_classes", type=int, default=3, help="BraTS 的分割类别数 (ET, TC, WT)")
    parser.add_argument("--input_channels", type=int, default=4, help="BraTS 输入模态数 (T1, T1ce, T2, FLAIR)")

    args = parser.parse_args()
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


def train_one_epoch(model, optimizer, train_loader, criterion, args):
    model.train()
    total_loss = 0.0
    for batched_input in train_loader:
        batched_input = to_device(batched_input, args.device)
        images = batched_input["image"]
        labels = batched_input["label"]

        with torch.no_grad():
            image_embeddings = model.image_encoder(images)

        accumulated_loss = 0
        for c in range(args.num_classes):
            boxes_c = batched_input["boxes"][:, c, :]
            labels_c = labels[:, c:c + 1, :, :]
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None, boxes=boxes_c.unsqueeze(1), masks=None,
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            upscaled_masks = F.interpolate(
                low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False,
            )
            loss = criterion(upscaled_masks, labels_c, iou_predictions)
            accumulated_loss += loss

        final_loss = accumulated_loss / args.num_classes
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        total_loss += final_loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, val_loader, args):
    model.eval()
    total_dice = np.zeros(args.num_classes)
    class_names = ["ET", "TC", "WT"]

    with torch.no_grad():
        for batched_input in val_loader:
            batched_input = to_device(batched_input, args.device)
            images = batched_input["image"]
            labels = batched_input["label"]
            image_embeddings = model.image_encoder(images)

            for c in range(args.num_classes):
                boxes_c = batched_input["boxes"][:, c, :]
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None, boxes=boxes_c.unsqueeze(1), masks=None,
                )
                low_res_masks, _ = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                upscaled_masks = F.interpolate(
                    low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False,
                )
                binary_masks = (torch.sigmoid(upscaled_masks) > 0.5).float()

                # 计算 Dice
                dice_c = SegMetrics(binary_masks, labels[:, c:c + 1, :, :], ['dice'])[0]
                total_dice[c] += dice_c

    avg_dice_per_class = total_dice / len(val_loader)
    avg_dice_mean = avg_dice_per_class.mean()  # 计算所有类别的平均Dice作为主要监控指标

    print(
        f"Validation Dice -> Mean: {avg_dice_mean:.4f}, ET: {avg_dice_per_class[0]:.4f}, TC: {avg_dice_per_class[1]:.4f}, WT: {avg_dice_per_class[2]:.4f}")
    return avg_dice_mean


def main(args):
    # --- 模型加载和修改 ---
    model_param_keys = ['model_type', 'sam_checkpoint', 'encoder_adapter', 'image_size']
    model_args_dict = {key: getattr(args, key) for key in model_param_keys if hasattr(args, key)}
    model_args = argparse.Namespace(**model_args_dict)
    model = sam_model_registry[args.model_type](model_args).to(args.device)
    original_proj = model.image_encoder.patch_embed.proj
    new_proj = nn.Conv2d(
        in_channels=args.input_channels, out_channels=original_proj.out_channels,
        kernel_size=original_proj.kernel_size, stride=original_proj.stride,
        padding=original_proj.padding, bias=(original_proj.bias is not None)
    ).to(args.device)
    with torch.no_grad():
        avg_weights = original_proj.weight.clone().detach().mean(dim=1, keepdim=True)
        new_proj.weight.copy_(avg_weights.repeat(1, args.input_channels, 1, 1))
        if original_proj.bias is not None: new_proj.bias.copy_(original_proj.bias)
    model.image_encoder.patch_embed.proj = new_proj

    # --- 应用 LoRA ---
    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=args.lora_target_modules,
                             lora_dropout=0.1, bias="none")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 优化器和损失函数 ---
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = FocalDiceloss_IoULoss()

    # --- 数据加载器 ---
    train_dataset = BraTSDataset(data_path=args.train_data_path, image_size=args.image_size,
                                 num_classes=args.num_classes, mode='train', subset_size=args.train_subset_size)
    val_dataset = BraTSDataset(data_path=args.val_data_path, image_size=args.image_size, num_classes=args.num_classes,
                               mode='val', subset_size=args.val_subset_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- 训练主循环 ---
    log_dir = os.path.join(args.work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(os.path.join(log_dir, f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    best_val_dice = -1.0
    patience_counter = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # 训练
        train_loss = train_one_epoch(model, optimizer, tqdm(train_loader, desc="Training"), criterion, args)
        logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")

        # 验证
        val_dice_mean = evaluate_model(model, tqdm(val_loader, desc="Validating"), args)
        logger.info(f"Epoch {epoch + 1}, Validation Mean Dice: {val_dice_mean:.4f}")

        # 保存最佳模型
        if val_dice_mean > best_val_dice:
            best_val_dice = val_dice_mean
            patience_counter = 0
            save_dir = os.path.join(args.work_dir, "models", args.run_name)
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            logger.info(f"New best model saved to {save_dir} with Mean Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Validation Dice did not improve. Patience: {patience_counter}/{args.early_stopping_patience}")

        # 检查早停
        if patience_counter >= args.early_stopping_patience:
            logger.info("Early stopping triggered.")
            break

    print(f"\nTraining finished. Best validation Mean Dice: {best_val_dice:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
