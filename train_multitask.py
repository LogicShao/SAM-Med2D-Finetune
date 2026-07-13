import argparse
import csv
import datetime
import os
import random
import time

# Must be present before importing Torch when deterministic CUDA matmul is requested.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the SimpleITK-backed dataset before PEFT to avoid a Windows DLL load
# conflict in local verification environments.
from multitask_dataset import BraTSDataset

from peft import get_peft_model, LoraConfig

from metrics import SegMetrics
from cli_utils import str_to_bool
from model_factory import build_multitask_base_model
from training_profiler import GpuUtilizationMonitor, parse_cuda_device_index
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
    parser.add_argument("--use_amp", type=str_to_bool, default=True, help="启用自动混合精度训练 (AMP)")
    parser.add_argument("--disable_cudnn", type=str_to_bool, default=False,
                        help="禁用 cuDNN，用于不稳定 CUDA 环境")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作进程数")
    parser.add_argument(
        "--non_blocking_transfer",
        type=str_to_bool,
        default=False,
        help="在 pinned memory 已启用时异步执行 CPU 到 GPU 的张量拷贝",
    )
    parser.add_argument(
        "--persistent_workers",
        type=str_to_bool,
        default=False,
        help="跨 epoch 保持 DataLoader worker，仅在 num_workers 大于零时有效",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=None,
        help="每个 DataLoader worker 预取 batch 数；默认使用 PyTorch 默认值",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="每个 epoch 最多执行的优化步数；默认使用全部训练数据。",
    )

    # --- 数据集和调试 ---
    parser.add_argument("--train_subset_size", type=int, default=None, help="使用训练集的前N个样本")
    parser.add_argument("--val_subset_size", type=int, default=None, help="使用验证集的前N个样本")
    parser.add_argument("--cache_root", type=str, default=None, help="预处理后的病例级 cache 根目录")
    parser.add_argument("--cache_max_cases", type=int, default=0, help="每个 DataLoader worker 持有的 memory-map 病例数")
    parser.add_argument("--negative_to_positive_ratio", type=float, default=0.0,
                        help="训练中每个阳性 slice 采样的阴性 slice 比例")
    parser.add_argument("--negative_prompt_box", choices=['zero', 'random'], default='zero',
                        help="阴性 slice 的 prompt box 策略")
    parser.add_argument("--dataset_seed", type=int, default=None,
                        help="负 slice 采样与随机阴性框种子；默认继承 --seed")
    parser.add_argument("--profile_performance", type=str_to_bool, default=False,
                        help="记录 DataLoader 等待和 CUDA event 计算时间")
    parser.add_argument("--profile_gpu_utilization", type=str_to_bool, default=False,
                        help="通过 nvidia-smi 后台采样 GPU 利用率")
    parser.add_argument("--profile_sample_interval", type=float, default=1.0,
                        help="GPU 利用率采样间隔，单位秒")
    parser.add_argument("--seed", type=int, default=42, help="全局训练随机种子")
    parser.add_argument("--deterministic", type=str_to_bool, default=False,
                        help="启用 Torch 最佳努力确定性模式；SAM CUDA cumsum 仍非位级确定")
    parser.add_argument("--save_epochs", type=int, nargs="*", default=[],
                        help="保存不可变 checkpoint 的 epoch 编号，例如 1 3 5")

    # --- 早停机制 ---
    parser.add_argument("--early_stopping_patience", type=int, default=20, help="验证集性能连续N轮不提升则早停")

    # --- LoRA/Adapter 参数 ---
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument('--lora_target_modules', nargs='+', default=['qkv'], help='应用 LoRA 的目标层')
    parser.add_argument("--encoder_adapter", type=str_to_bool, default=True, help="模型是否包含 Adapter (用于构建和冻结)")

    # --- BraTS 特定的参数定义 ---
    parser.add_argument("--num_classes", type=int, default=3, help="BraTS 的分割类别数 (ET, TC, WT)")
    parser.add_argument("--input_channels", type=int, default=4, help="BraTS 输入模态数")

    args = parser.parse_args()
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        parser.error("--max_train_steps must be greater than zero when provided.")
    if args.cache_max_cases < 0:
        parser.error("--cache_max_cases must be zero or greater.")
    if args.num_workers < 0:
        parser.error("--num_workers must be zero or greater.")
    if args.prefetch_factor is not None and args.prefetch_factor <= 0:
        parser.error("--prefetch_factor must be greater than zero when provided.")
    if args.num_workers == 0 and args.persistent_workers:
        parser.error("--persistent_workers requires --num_workers greater than zero.")
    if args.num_workers == 0 and args.prefetch_factor is not None:
        parser.error("--prefetch_factor requires --num_workers greater than zero.")
    if args.negative_to_positive_ratio < 0.0:
        parser.error("--negative_to_positive_ratio must be zero or greater.")
    if args.profile_sample_interval <= 0.0:
        parser.error("--profile_sample_interval must be greater than zero.")
    if any(epoch <= 0 for epoch in args.save_epochs):
        parser.error("--save_epochs values must be greater than zero.")
    args.save_epochs = sorted(set(args.save_epochs))
    if args.dataset_seed is None:
        args.dataset_seed = args.seed
    args.run_name = f"{args.run_name}_{args.finetune_method}"
    return args


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
    worker_info = torch.utils.data.get_worker_info()
    transform = getattr(worker_info.dataset, "transform", None) if worker_info is not None else None
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(worker_seed)


def save_epoch_snapshot(model, args, model_dir, epoch, logger):
    if epoch not in args.save_epochs:
        return

    if args.finetune_method == 'lora':
        snapshot_path = os.path.join(model_dir, 'epoch_{:03d}_lora_adapters'.format(epoch))
        if os.path.exists(snapshot_path):
            raise FileExistsError("Refusing to overwrite immutable epoch snapshot: {}".format(snapshot_path))
        model.image_encoder.save_pretrained(snapshot_path)
    else:
        snapshot_path = os.path.join(model_dir, 'epoch_{:03d}.pth'.format(epoch))
        if os.path.exists(snapshot_path):
            raise FileExistsError("Refusing to overwrite immutable epoch snapshot: {}".format(snapshot_path))
        torch.save(model.state_dict(), snapshot_path)
    logger.info("Saved immutable epoch snapshot: %s", snapshot_path)


def to_device(batch_input, device, non_blocking=False):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key in ['image', 'label']:
                device_input[key] = value.float().to(device, non_blocking=non_blocking)
            elif isinstance(value, (list, torch.Size)):
                device_input[key] = value
            else:
                device_input[key] = value.to(device, non_blocking=non_blocking)
        else:
            device_input[key] = value
    return device_input


def train_one_epoch(model, optimizer, train_loader, criterion, args, epoch, scaler):
    model.train()
    total_loss = 0.0
    total_dice = np.zeros(args.num_classes)
    total_iou = np.zeros(args.num_classes)
    step_count = 0
    sample_count = 0
    data_wait_seconds = []
    host_to_device_seconds = []
    gpu_compute_seconds = []
    gpu_monitor = None
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.profile_gpu_utilization:
        gpu_index = parse_cuda_device_index(args.device)
        if gpu_index is not None:
            gpu_monitor = GpuUtilizationMonitor(gpu_index, args.profile_sample_interval)
            gpu_monitor.start()
    start_time = time.perf_counter()
    previous_step_end = start_time

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
    for batched_input in pbar:
        data_wait_seconds.append(time.perf_counter() - previous_step_end)
        transfer_start = None
        transfer_end = None
        if args.profile_performance and str(args.device).startswith("cuda"):
            transfer_start = torch.cuda.Event(enable_timing=True)
            transfer_end = torch.cuda.Event(enable_timing=True)
            transfer_start.record()
        batched_input = to_device(
            batched_input,
            args.device,
            non_blocking=args.non_blocking_transfer and str(args.device).startswith("cuda"),
        )
        if transfer_end is not None:
            transfer_end.record()
        images, labels = batched_input["image"], batched_input["label"]
        sample_count += int(images.shape[0])

        compute_start = None
        compute_end = None
        if args.profile_performance and str(args.device).startswith("cuda"):
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record()

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
        if compute_end is not None:
            compute_end.record()
            compute_end.synchronize()
            host_to_device_seconds.append(transfer_start.elapsed_time(transfer_end) / 1000.0)
            gpu_compute_seconds.append(compute_start.elapsed_time(compute_end) / 1000.0)

        loss_item = final_loss.item()
        total_loss += loss_item
        pbar.set_postfix(loss=f'{loss_item:.4f}')

        with torch.no_grad():
            for c in range(args.num_classes):
                binary_masks = (torch.sigmoid(all_class_masks[c]) > 0.5).float()
                dice_c, iou_c = SegMetrics(binary_masks, labels[:, c:c + 1, :, :], ['dice', 'iou'])
                total_dice[c] += dice_c
                total_iou[c] += iou_c

        step_count += 1
        previous_step_end = time.perf_counter()
        if args.max_train_steps is not None and step_count >= args.max_train_steps:
            break

    if step_count == 0:
        raise ValueError("No training steps were executed. Check the dataset and --max_train_steps.")

    elapsed_seconds = time.perf_counter() - start_time
    runtime_stats = {
        "steps": step_count,
        "samples": sample_count,
        "elapsed_seconds": elapsed_seconds,
        "updates_per_second": step_count / elapsed_seconds if elapsed_seconds > 0.0 else None,
        "samples_per_second": sample_count / elapsed_seconds if elapsed_seconds > 0.0 else None,
        "peak_memory_allocated_mib": None,
        "peak_memory_reserved_mib": None,
        "mean_data_wait_seconds": float(np.mean(data_wait_seconds)) if data_wait_seconds else None,
        "mean_host_to_device_seconds": float(np.mean(host_to_device_seconds)) if host_to_device_seconds else None,
        "mean_gpu_compute_seconds": float(np.mean(gpu_compute_seconds)) if gpu_compute_seconds else None,
    }
    if str(args.device).startswith("cuda"):
        runtime_stats["peak_memory_allocated_mib"] = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)
        runtime_stats["peak_memory_reserved_mib"] = torch.cuda.max_memory_reserved(args.device) / (1024 ** 2)
    runtime_stats["gpu_utilization"] = gpu_monitor.stop() if gpu_monitor is not None else {"enabled": False}

    avg_loss = total_loss / step_count
    avg_dice = total_dice / step_count
    avg_iou = total_iou / step_count
    return avg_loss, avg_dice, avg_iou, runtime_stats


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
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    logger = get_logger(os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.log"))

    logger.info("--- Command Line Arguments ---")
    for key, value in vars(args).items(): logger.info(f"{key}: {value}")
    logger.info("----------------------------\n")

    seed_everything(args.seed, args.deterministic)
    logger.info(
        "Global seed: %d | dataset seed: %d | deterministic: %s | CUBLAS_WORKSPACE_CONFIG: %s",
        args.seed,
        args.dataset_seed,
        args.deterministic,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    is_amp_enabled = args.use_amp and ('cuda' in args.device)
    args.use_amp = is_amp_enabled
    scaler = GradScaler(enabled=is_amp_enabled)
    logger.info(f"Automatic Mixed Precision (AMP) enabled: {is_amp_enabled}")
    logger.info(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    model = build_multitask_base_model(
        model_type=args.model_type,
        image_size=args.image_size,
        sam_checkpoint=args.sam_checkpoint,
        input_channels=args.input_channels,
        encoder_adapter=args.encoder_adapter,
    ).to(args.device)
    logger.info("模型已加载预训练权重并适配为 %d 通道输入。", args.input_channels)
    print(f"模型已加载预训练权重并适配为 {args.input_channels} 通道输入。")

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
                                 num_classes=args.num_classes, mode='train', subset_size=args.train_subset_size,
                                 cache_root=args.cache_root, cache_max_cases=args.cache_max_cases,
                                 negative_to_positive_ratio=args.negative_to_positive_ratio,
                                 negative_prompt_box=args.negative_prompt_box, sample_seed=args.dataset_seed)
    val_dataset = BraTSDataset(data_path=args.val_data_path, image_size=args.image_size, num_classes=args.num_classes,
                               mode='val', subset_size=args.val_subset_size, cache_root=args.cache_root,
                               cache_max_cases=args.cache_max_cases, negative_to_positive_ratio=0.0,
                               negative_prompt_box='zero', sample_seed=args.dataset_seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
        persistent_workers=args.persistent_workers,
        **({"prefetch_factor": args.prefetch_factor} if args.prefetch_factor is not None else {}),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
        persistent_workers=args.persistent_workers,
        **({"prefetch_factor": args.prefetch_factor} if args.prefetch_factor is not None else {}),
    )

    # --- 统一的日志和 CSV 记录 ---
    csv_path = os.path.join(log_dir, "metrics.csv")
    csv_header = ['epoch', 'train_steps', 'train_samples', 'train_elapsed_seconds', 'train_updates_per_second',
                  'samples_per_second', 'peak_memory_allocated_mib', 'peak_memory_reserved_mib', 'train_loss', 'val_loss',
                  'mean_data_wait_seconds', 'mean_host_to_device_seconds', 'mean_gpu_compute_seconds',
                  'gpu_utilization_mean_percent',
                  'train_dice_mean', 'val_dice_mean', 'train_iou_mean', 'val_iou_mean',
                  'val_dice_ET', 'val_dice_TC', 'val_dice_WT', 'val_iou_ET', 'val_iou_TC', 'val_iou_WT']
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_header)

    best_val_dice = -1.0
    patience_counter = 0

    print(f"\n开始训练... 日志文件位于: {log_dir}")
    for epoch in range(args.epochs):
        train_loss, train_dice, train_iou, runtime_stats = train_one_epoch(
            model, optimizer, train_loader, criterion, args, epoch, scaler
        )
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, args)

        train_dice_mean, train_iou_mean = train_dice.mean(), train_iou.mean()
        val_dice_mean, val_iou_mean = val_dice.mean(), val_iou.mean()

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | Val Dice: {val_dice_mean:.4f} | ET: {val_dice[0]:.4f}, TC: {val_dice[1]:.4f}, WT: {val_dice[2]:.4f}")
        logger.info(f"Epoch {epoch + 1} | Train -> Loss: {train_loss:.4f}, Mean Dice: {train_dice_mean:.4f}, Mean IoU: {train_iou_mean:.4f}")
        logger.info(
            "Epoch %d | Runtime -> steps: %d, elapsed: %.2fs, updates/s: %.4f, samples/s: %.4f, "
            "peak allocated: %.1f MiB, peak reserved: %.1f MiB",
            epoch + 1,
            runtime_stats["steps"],
            runtime_stats["elapsed_seconds"],
            runtime_stats["updates_per_second"],
            runtime_stats["samples_per_second"],
            runtime_stats["peak_memory_allocated_mib"] or 0.0,
            runtime_stats["peak_memory_reserved_mib"] or 0.0,
        )
        logger.info(
            "Epoch %d | Profile -> CPU batch wait: %s, H2D: %s, GPU compute: %s, GPU utilization: %s",
            epoch + 1,
            runtime_stats["mean_data_wait_seconds"],
            runtime_stats["mean_host_to_device_seconds"],
            runtime_stats["mean_gpu_compute_seconds"],
            runtime_stats["gpu_utilization"],
        )
        logger.info(f"Epoch {epoch + 1} | Val   -> Loss: {val_loss:.4f}, Mean Dice: {val_dice_mean:.4f}, Mean IoU: {val_iou_mean:.4f}")
        logger.info(f"Val Dice per class -> ET: {val_dice[0]:.4f}, TC: {val_dice[1]:.4f}, WT: {val_dice[2]:.4f}")
        logger.info(f"Val IoU per class  -> ET: {val_iou[0]:.4f}, TC: {val_iou[1]:.4f}, WT: {val_iou[2]:.4f}\n")

        row_data = [
            epoch + 1,
            runtime_stats["steps"],
            runtime_stats["samples"],
            runtime_stats["elapsed_seconds"],
            runtime_stats["updates_per_second"],
            runtime_stats["samples_per_second"],
            runtime_stats["peak_memory_allocated_mib"],
            runtime_stats["peak_memory_reserved_mib"],
            train_loss,
            val_loss,
            runtime_stats["mean_data_wait_seconds"],
            runtime_stats["mean_host_to_device_seconds"],
            runtime_stats["mean_gpu_compute_seconds"],
            runtime_stats["gpu_utilization"].get("gpu_utilization_percent_mean"),
            train_dice_mean,
            val_dice_mean,
            train_iou_mean,
            val_iou_mean,
            val_dice[0],
            val_dice[1],
            val_dice[2],
            val_iou[0],
            val_iou[1],
            val_iou[2],
        ]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([f'{v:.4f}' if isinstance(v, (float, np.floating)) else v for v in row_data])

        save_epoch_snapshot(model, args, model_dir, epoch + 1, logger)

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
