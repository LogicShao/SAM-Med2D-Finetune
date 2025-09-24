import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from BraTSDataset import BraTSDataset
from metrics import SegMetrics
from segment_anything import sam_model_registry

os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM-Med2D on BraTS 2021 (Fine-tuned vs. Pre-trained)")

    # --- 路径和模型配置 ---
    parser.add_argument("--data_path", type=str, required=True, help="BraTS 2021 验证集路径")
    parser.add_argument("--work_dir", type=str, default="workdir_brats", help="工作目录，用于查找微调后的模型")
    parser.add_argument("--run_name", type=str, default="sam-med2d-brats-lora", help="微调模型的运行名称")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM 模型类型")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth",
                        help="原始的 SAM-Med2D 预训练权重路径")

    # --- 评估参数 ---
    parser.add_argument("--batch_size", type=int, default=1, help="评估批次大小")
    parser.add_argument("--image_size", type=int, default=256, help="输入图像尺寸")
    parser.add_argument('--device', type=str, default='cuda', help="使用的设备")
    parser.add_argument("--metrics", nargs='+', default=['dice', 'iou'], help="计算的指标")
    parser.add_argument("--subset_size", type=int, default=None, help="为快速评估，限制验证集为前N个样本")

    # --- BraTS 特定参数 ---
    parser.add_argument("--num_classes", type=int, default=3, help="BraTS 的分割类别数")
    parser.add_argument("--input_channels", type=int, default=4, help="BraTS 输入模态数")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="加载模型时是否构建 Adapter 层")

    # --- 关键修改：新增用于对比的开关 ---
    parser.add_argument("--eval_pretrained_only", action='store_true',
                        help="如果设置此项，将只评估原始预训练模型，不加载LoRA权重。")

    args = parser.parse_args()
    return args


def to_device(batch_input, device):
    # ... (此函数保持不变)
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


def main(args):
    # --- 1. 加载基础模型 (所有评估流程都需要的步骤) ---
    print("加载 SAM-Med2D 基础模型...")
    model_args = argparse.Namespace(**vars(args))
    model = sam_model_registry[args.model_type](model_args).to(args.device)

    # --- 2. 适配 BraTS 输入 (所有评估流程都需要的步骤) ---
    original_proj = model.image_encoder.patch_embed.proj
    new_proj = nn.Conv2d(
        in_channels=args.input_channels,
        out_channels=original_proj.out_channels,
        kernel_size=original_proj.kernel_size,
        stride=original_proj.stride,
        padding=original_proj.padding,
        bias=(original_proj.bias is not None)
    ).to(args.device)
    model.image_encoder.patch_embed.proj = new_proj
    print("模型输入层已适配为 4 通道。")

    # --- 3. 关键修改：根据开关决定是否加载 LoRA 权重 ---
    if args.eval_pretrained_only:
        print("\n--- 模式: 仅评估预训练模型 (不加载 LoRA 权重) ---\n")
    else:
        print("\n--- 模式: 评估微调后的模型 (加载 LoRA 权重) ---\n")
        lora_weights_path = os.path.join(args.work_dir, "models", args.run_name)
        absolute_lora_path = os.path.abspath(lora_weights_path)

        if not os.path.exists(absolute_lora_path):
            raise FileNotFoundError(f"LoRA 权重目录未找到: {absolute_lora_path}")

        print(f"从以下路径加载 LoRA 适配器: {absolute_lora_path}")
        model = PeftModel.from_pretrained(model, absolute_lora_path, is_trainable=False)
        print("LoRA 适配器加载成功。")

    # --- 4. 准备验证数据集 (所有评估流程都需要的步骤) ---
    val_dataset = BraTSDataset(
        data_path=args.data_path,
        image_size=args.image_size,
        num_classes=args.num_classes,
        mode='val',
        subset_size=args.subset_size
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f'验证数据集大小: {len(val_dataset)}')

    # --- 5. 评估循环 (逻辑保持不变) ---
    model.eval()
    total_metrics = np.zeros(args.num_classes * len(args.metrics))
    class_names = ["ET", "TC", "WT"]

    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc="正在评估")
        for batched_input in val_loader_tqdm:
            batched_input = to_device(batched_input, args.device)
            images = batched_input["image"]
            labels = batched_input["label"]

            image_embeddings = model.image_encoder(images)

            all_pred_masks = []
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
                all_pred_masks.append(binary_masks)

            all_pred_masks = torch.cat(all_pred_masks, dim=1)

            batch_metrics = []
            for c in range(args.num_classes):
                pred_c = all_pred_masks[:, c:c + 1, :, :]
                label_c = labels[:, c:c + 1, :, :]
                metrics_c = SegMetrics(pred_c, label_c, args.metrics)
                batch_metrics.extend(metrics_c)

            total_metrics += np.array(batch_metrics)

    # --- 6. 聚合和显示结果 (逻辑保持不变) ---
    avg_metrics = total_metrics / len(val_loader)

    print("\n--- 评估结果 ---")
    i = 0
    for c in range(args.num_classes):
        print(f"类别: {class_names[c]}")
        for metric_name in args.metrics:
            print(f"  - 平均 {metric_name}: {avg_metrics[i]:.4f}")
            i += 1
    print("------------------\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)
