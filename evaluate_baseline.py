import argparse
import json
import os

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader import SAMDataset
from metrics import SegMetrics
from segment_anything import sam_model_registry
from utils import FocalDiceloss_IoULoss, get_logger


def evaluate(model, val_loader, criterion, args):
    """
    在验证集上评估模型性能。

    Args:
        model (torch.nn.Module): 要评估的模型。
        val_loader (DataLoader): 验证数据加载器。
        criterion (torch.nn.Module): 损失函数。
        args (argparse.Namespace): 命令行参数。

    Returns:
        tuple: 包含平均损失、平均 Dice 和平均 IoU 的元组。
    """
    model.eval()
    total_loss = 0.0
    # 使用 numpy 数组累加指标，更高效
    total_dice = []
    total_iou = []

    # 使用 torch.no_grad() 禁用梯度计算，加速推理并节省显存
    with torch.no_grad():
        # 使用 tqdm 创建一个可视化的进度条
        pbar = tqdm(val_loader, desc="正在评估基线模型")
        for batched_input in pbar:
            # 跳过数据加载失败的样本
            if batched_input is None:
                continue

            # 确保数据类型正确
            images = batched_input["image"].to(args.device, dtype=torch.float32)
            labels = batched_input["label"].to(args.device)
            boxes = batched_input["boxes"].to(args.device)

            # 使用混合精度 (AMP) 加速推理
            with autocast(device_type=args.device.split(':')[0], enabled=args.use_amp):
                image_embeddings = model.image_encoder(images)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=boxes, masks=None)

                low_res_masks, iou_predictions = model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = torch.nn.functional.interpolate(
                    low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False)

                loss = criterion(upscaled_masks, labels, iou_predictions)

            total_loss += loss.item()

            # 将预测掩码转换为二值图 (0或1)
            binary_masks = (torch.sigmoid(upscaled_masks) > 0.5).float()

            # 计算当前批次的指标
            dice, iou = SegMetrics(binary_masks, labels, ['dice', 'iou'])
            total_dice.append(dice)
            total_iou.append(iou)

            # 在进度条上实时更新平均指标
            pbar.set_postfix(
                loss=f'{total_loss / len(total_dice):.4f}',
                dice=f'{np.mean(total_dice):.4f}',
                iou=f'{np.mean(total_iou):.4f}'
            )

    # 计算最终的平均指标
    avg_loss = total_loss / len(val_loader)
    avg_dice = np.mean(total_dice)
    avg_iou = np.mean(total_iou)

    return avg_loss, avg_dice, avg_iou


def main(args):
    # 配置日志记录器
    os.makedirs(args.work_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.work_dir, 'baseline_evaluation.log'))
    logger.info("开始评估基线模型 (无微调)...")
    logger.info(f"参数: {vars(args)}")

    # 1. 加载模型
    print("正在加载原始 SAM-Med2D 模型...")
    # 同样地，我们将模型创建和权重加载分开
    model = sam_model_registry[args.model_type](args).to(args.device)
    if args.sam_checkpoint and os.path.isfile(args.sam_checkpoint):
        try:
            with open(args.sam_checkpoint, "rb") as f:
                state_dict = torch.load(f, map_location=args.device, weights_only=False)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                model.load_state_dict(state_dict, strict=False)
            print(f"成功加载预训练权重: {args.sam_checkpoint}")
            logger.info(f"成功加载预训练权重: {args.sam_checkpoint}")
        except Exception as e:
            print(f"加载权重失败: {e}")
            logger.error(f"加载权重失败: {e}")
            return  # 如果权重加载失败，则无法进行评估
    else:
        print("错误：找不到指定的模型权重文件。")
        logger.error("错误：找不到指定的模型权重文件。")
        return

    # 2. 初始化损失函数
    criterion = FocalDiceloss_IoULoss()

    # 3. 准备数据加载器
    print("正在准备验证数据集...")
    val_dataset = SAMDataset(data_path=args.data_path, image_size=args.image_size, mode='val')
    # 推荐 batch_size=1，但允许通过命令行修改
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if len(val_dataset) == 0:
        print("错误：在指定路径下没有找到验证数据。")
        logger.error("错误：在指定路径下没有找到验证数据。")
        return

    # 4. 执行评估
    avg_loss, avg_dice, avg_iou = evaluate(model, val_loader, criterion, args)

    # 5. 打印并保存结果
    print("\n" + "=" * 30)
    print("--- 基线模型评估结果 ---")
    print(f"  > 平均验证损失 (Loss): {avg_loss:.4f}")
    print(f"  > 平均验证 Dice:        {avg_dice:.4f}")
    print(f"  > 平均验证 IoU:         {avg_iou:.4f}")
    print("=" * 30 + "\n")

    results = {'avg_loss': avg_loss, 'avg_dice': avg_dice, 'avg_iou': avg_iou}
    results = {k: float(v) for k, v in results.items()}
    os.makedirs(args.work_dir, exist_ok=True)
    result_path = os.path.join(args.work_dir, 'baseline_metrics.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"评估结果已保存到: {result_path}")
    logger.info(f"评估结果已保存到: {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估原始 SAM-Med2D 模型在指定数据集上的基线性能")
    parser.add_argument("--data_path", type=str, default="data_brats_processed", help="预处理后的数据路径")
    parser.add_argument("--work_dir", type=str, default="workdir_brats", help="用于保存日志和结果的工作目录")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--batch_size", type=int, default=1, help="评估时的批次大小 (推荐为 1)")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM 模型类型")
    parser.add_argument("--sam_checkpoint", type=str, default="pretrain_model/sam-med2d_b.pth",
                        help="预训练 SAM-Med2D 权重路径")
    parser.add_argument('--device', type=str, default='cuda', help="设备 (e.g., 'cuda', 'cpu')")
    parser.add_argument("--use_amp", action='store_true', help="在评估时启用混合精度以加速", default=True)
    parser.add_argument("--encoder_adapter", type=bool, default=True,
                        help="模型是否包含 Adapter 层 (与 SAM-Med2D 保持一致)")

    args = parser.parse_args()
    main(args)
