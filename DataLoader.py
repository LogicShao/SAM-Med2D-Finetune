import json
import logging
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import train_transforms, get_boxes_from_mask, init_point_sampling

logging.basicConfig(level=logging.INFO)


class SAMDataset(Dataset):
    """
    一个统一的数据集类，用于处理 SAM-Med2D 的训练、验证和测试数据。
    通过 'mode' 参数来区分不同的数据处理流程。
    """

    def __init__(self, data_path, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1,
                 return_ori_mask=True):
        """
        初始化 SAMDataset 对象。

        Args:
            data_path (str): 数据集根目录，应包含 JSON 映射文件。
            image_size (int): 图像的目标尺寸。
            mode (str): 数据集模式，'train', 'val', 或 'test'。
            requires_name (bool): 是否返回样本名称。
            point_num (int): 每个掩码采样的点提示数量。
            mask_num (int): 每个图像采样的掩码数量（仅在 'train' 模式下有效）。
            return_ori_mask (bool): 是否返回原始（未变换的）掩码（仅在 'val'/'test' 模式下有效）。
        """
        self.image_size = image_size
        self.mode = mode
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.return_ori_mask = return_ori_mask

        # 使用 with 语句安全地打开文件
        if self.mode == 'train':
            json_path = os.path.join(data_path, 'image2label_train.json')
            with open(json_path, 'r') as f:
                dataset = json.load(f)
            self.image_paths = list(dataset.keys())
            self.label_groups = list(dataset.values())  # 每个图像对应一组掩码路径
        else:  # 'val' 或 'test' 模式
            json_path = os.path.join(data_path, f'label2image_{mode}.json')
            with open(json_path, 'r') as f:
                dataset = json.load(f)
            self.label_paths = list(dataset.keys())
            self.image_paths = list(dataset.values())

        # 转换为 Numpy 数组以便高效计算
        self.pixel_mean = np.array([123.675, 116.28, 103.53]).reshape(1, 1, -1)
        self.pixel_std = np.array([58.395, 57.12, 57.375]).reshape(1, 1, -1)

        logging.info(f"成功加载 {len(self)} 个样本，模式: {self.mode}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image_input = {}

        try:
            image = cv2.imread(image_path)
            if image is None: raise IOError(f"无法读取图像文件: {image_path}")
            image = image.astype(np.float32)
            image = (image - self.pixel_mean) / self.pixel_std
        except Exception as e:
            logging.error(f"加载或预处理图像 {image_path} 时出错: {e}")
            return None

        if self.mode == 'train':
            h, w, _ = image.shape
            transforms = train_transforms(self.image_size, h, w)

            # --- 核心修正：确保 image 和 masks 的数量一致 ---
            images_list, masks_list, boxes_list = [], [], []
            point_coords_list, point_labels_list = [], []

            selected_mask_paths = random.choices(self.label_groups[index], k=self.mask_num)

            for mask_path in selected_mask_paths:
                try:
                    pre_mask = cv2.imread(mask_path, 0).astype(np.float32)
                    if pre_mask.max() == 255.0: pre_mask /= 255.0

                    augments = transforms(image=image, mask=pre_mask)
                    image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

                    boxes = get_boxes_from_mask(mask_tensor)
                    point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

                    # 为每个掩码都添加一个对应的图像
                    images_list.append(image_tensor)
                    masks_list.append(mask_tensor.unsqueeze(0))  # 保持通道维度
                    boxes_list.append(boxes)
                    point_coords_list.append(point_coords)
                    point_labels_list.append(point_label)
                except Exception as e:
                    logging.warning(f"处理掩码 {mask_path} 时出错: {e}，跳过此掩码。")
                    continue

            if not masks_list:
                return None

            # 现在，每个张量列表的长度都是 mask_num
            # DataLoader 的默认 collate_fn 会将它们打包成 [B, mask_num, ...]
            image_input["image"] = torch.stack(images_list)
            image_input["label"] = torch.stack(masks_list)
            image_input["boxes"] = torch.stack(boxes_list)
            image_input["point_coords"] = torch.stack(point_coords_list)
            image_input["point_labels"] = torch.stack(point_labels_list)

        else:  # 'val'/'test' 模式逻辑不变
            mask_path = self.label_paths[index]
            try:
                ori_np_mask = cv2.imread(mask_path, 0).astype(np.float32)
                if ori_np_mask.max() == 255.0: ori_np_mask /= 255.0
            except Exception as e:
                logging.error(f"加载或处理掩码 {mask_path} 时出错: {e}")
                return None

            h, w = ori_np_mask.shape
            transforms = train_transforms(self.image_size, h, w)
            augments = transforms(image=image, mask=ori_np_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_labels = init_point_sampling(mask_tensor, self.point_num)

            image_input["image"] = image_tensor
            image_input["label"] = mask_tensor.unsqueeze(0)
            image_input["boxes"] = boxes
            image_input["point_coords"] = point_coords
            image_input["point_labels"] = point_labels
            image_input["original_size"] = (h, w)
            if self.return_ori_mask:
                image_input["ori_label"] = torch.from_numpy(ori_np_mask).unsqueeze(0)

        if self.requires_name:
            image_input["name"] = os.path.basename(image_path)

        return image_input


def custom_collate_fn(batch):
    """
    自定义 Collate Function。
    对于训练模式，将 [B, mask_num, ...] 的张量展平为 [B * mask_num, ...]。
    对于验证模式，行为与默认 collate_fn 相同。
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 检查是否为训练模式（通过判断 image 维度）
    is_train = batch[0]['image'].dim() == 4  # Train item: [mask_num, C, H, W]

    collated_batch = torch.utils.data.default_collate(batch)

    if is_train:
        # 展平 batch 和 mask_num 维度
        for key in ['image', 'label', 'boxes', 'point_coords', 'point_labels']:
            if key in collated_batch:
                tensor = collated_batch[key]
                collated_batch[key] = tensor.reshape(-1, *tensor.shape[2:])

    return collated_batch


if __name__ == "__main__":
    # --- 演示如何使用新的统一数据集 ---
    DATA_DEMO_PATH = "data_brats_processed"  # 替换为你的数据路径

    print("\n--- 测试训练模式 ---")
    # 实例化训练数据集
    train_dataset = SAMDataset(
        DATA_DEMO_PATH,
        image_size=256,
        mode='train',
        mask_num=5,  # 训练时每个图像采样5个掩码
        point_num=1
    )
    # 训练 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn  # 训练时需要自定义 collate_fn
    )

    for i, batch in enumerate(tqdm(train_loader, desc="测试训练加载器")):
        if batch is None: continue
        # 经过 stack_dict_batched 处理后，形状应为 [B * mask_num, C, H, W]
        print(f"\n批次 {i + 1}:")
        print(f"  图像形状: {batch['image'].shape}")  # 预期: [2*5, 3, 256, 256]
        print(f"  标签形状: {batch['label'].shape}")  # 预期: [2*5, 1, 256, 256]
        print(f"  Boxes 形状: {batch['boxes'].shape}")  # 预期: [2*5, 1, 4]
        if i >= 0: break  # 只测试一个批次

    print("\n\n--- 测试验证模式 ---")
    # 实例化验证数据集
    val_dataset = SAMDataset(
        DATA_DEMO_PATH,
        image_size=256,
        mode='val',
        requires_name=True
    )
    # 验证 DataLoader (使用默认 collate_fn)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )

    for i, batch in enumerate(tqdm(val_loader, desc="测试验证加载器")):
        if batch is None: continue
        print(f"\n批次 {i + 1}:")
        print(f"  图像形状: {batch['image'].shape}")  # 预期: [4, 3, 256, 256]
        print(f"  标签形状: {batch['label'].shape}")  # 预期: [4, 1, 256, 256]
        print(f"  Boxes 形状: {batch['boxes'].shape}")  # 预期: [4, 1, 4]
        print(f"  样本名称: {batch['name']}")
        if i >= 0: break  # 只测试一个批次
