import os
from collections import OrderedDict
from pathlib import Path

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import SimpleITK as sitk
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from sam_med2d_finetune.brats.cache import (
    CACHE_IMAGES_FILENAME,
    CACHE_SEGMENTATION_FILENAME,
    validate_cache_case,
)

def get_main_bounding_box(mask_tensor):
    y_indices, x_indices = torch.where(mask_tensor > 0)
    if len(y_indices) == 0:
        return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    x_min, x_max = torch.min(x_indices), torch.max(x_indices)
    y_min, y_max = torch.min(y_indices), torch.max(y_indices)
    return torch.tensor([x_min.item(), y_min.item(), x_max.item(), y_max.item()], dtype=torch.float32)


class BraTSDataset(Dataset):
    def __init__(
        self,
        data_path,
        image_size=256,
        num_classes=3,
        mode='train',
        subset_size=None,
        cache_root=None,
        cache_max_cases=0,
        negative_to_positive_ratio=0.0,
        negative_prompt_box='zero',
        sample_seed=42,
        pjt_enabled=False,
        pjt_translate_max=0.10,
        pjt_scale_min=0.85,
        pjt_scale_max=1.15,
        pjt_miss_prob=0.0,
        pjt_seed=None,
    ):
        self.data_path = data_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.mode = mode
        self.cache_root = Path(cache_root) if cache_root else None
        self.cache_max_cases = max(int(cache_max_cases), 0)
        self.negative_to_positive_ratio = max(float(negative_to_positive_ratio), 0.0)
        self.negative_prompt_box = str(negative_prompt_box)
        self.sample_seed = int(sample_seed)
        self.pjt_enabled = bool(pjt_enabled)
        self.pjt_translate_max = max(0.0, float(pjt_translate_max))
        self.pjt_scale_min = min(float(pjt_scale_min), float(pjt_scale_max))
        self.pjt_scale_max = max(float(pjt_scale_min), float(pjt_scale_max))
        self.pjt_miss_prob = max(0.0, min(1.0, float(pjt_miss_prob)))
        self.pjt_seed = int(pjt_seed) if pjt_seed is not None else int(sample_seed)
        self._case_cache = OrderedDict()
        self._cache_metadata = {}

        if self.negative_prompt_box not in {'zero', 'random'}:
            raise ValueError("negative_prompt_box must be 'zero' or 'random'.")

        self.patients = sorted([p for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))])
        if self.cache_root is not None:
            for patient_id in self.patients:
                self._cache_metadata[patient_id] = validate_cache_case(self.cache_root, patient_id)["metadata"]

        self.slice_list = []
        positive_count = 0
        negative_count = 0
        rng = np.random.default_rng(self.sample_seed)
        for patient_id in self.patients:
            seg_vol = self._load_segmentation_for_index(patient_id)
            positive_indices = np.flatnonzero(np.any(seg_vol > 0, axis=(1, 2))).tolist()
            negative_indices = np.flatnonzero(~np.any(seg_vol > 0, axis=(1, 2))).tolist()
            self.slice_list.extend((patient_id, int(slice_idx)) for slice_idx in positive_indices)
            positive_count += len(positive_indices)

            selected_negative_count = min(
                len(negative_indices),
                int(round(len(positive_indices) * self.negative_to_positive_ratio)),
            )
            if selected_negative_count > 0:
                selected_negative_indices = rng.choice(negative_indices, size=selected_negative_count, replace=False)
                self.slice_list.extend((patient_id, int(slice_idx)) for slice_idx in selected_negative_indices)
                negative_count += selected_negative_count

        if subset_size is not None:
            self.slice_list = self.slice_list[:subset_size]
        print(
            f"Found {len(self.slice_list)} slices for {self.mode} set "
            f"(source_positive={positive_count}, source_negative={negative_count}, "
            f"cache={'enabled' if self.cache_root else 'disabled'})."
        )

        # --- 数据增强 Transform ---
        if self.mode == 'train':
            # 为训练集定义强大的数据增强
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                # --- 关键修改：用 Affine 替换 ShiftScaleRotate ---
                A.Affine(
                    translate_percent={'x': (-0.06, 0.06), 'y': (-0.06, 0.06)},  # 对应 shift_limit
                    scale=(1 - 0.1, 1 + 0.1),  # 对应 scale_limit
                    rotate=(-15, 15),  # 对应 rotate_limit
                    p=0.7,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=0,
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.ElasticTransform(p=0.5, border_mode=cv2.BORDER_CONSTANT),
            ])
        else:
            # 验证集通常不需要数据增强
            self.transform = None

    def __len__(self):
        return len(self.slice_list)

    def _cache_paths(self, patient_id):
        if self.cache_root is None:
            return None, None
        case_cache_dir = self.cache_root / patient_id
        return case_cache_dir / CACHE_IMAGES_FILENAME, case_cache_dir / CACHE_SEGMENTATION_FILENAME

    def _load_segmentation_for_index(self, patient_id):
        _, cached_seg_path = self._cache_paths(patient_id)
        if cached_seg_path is not None:
            if not cached_seg_path.is_file():
                raise FileNotFoundError(f"Missing cached segmentation for {patient_id}: {cached_seg_path}")
            return np.load(str(cached_seg_path), mmap_mode='r')

        patient_folder = os.path.join(self.data_path, patient_id)
        seg_path = os.path.join(patient_folder, f'{patient_id}_seg.nii.gz')
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Missing segmentation for {patient_id}: {seg_path}")
        return sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

    def _load_cached_case(self, patient_id):
        if patient_id in self._case_cache:
            self._case_cache.move_to_end(patient_id)
            return self._case_cache[patient_id]

        images_path, segmentation_path = self._cache_paths(patient_id)
        if images_path is None or not images_path.is_file() or not segmentation_path.is_file():
            raise FileNotFoundError(f"Missing cached case files for {patient_id} under {self.cache_root}")
        cached_case = {
            'images': np.load(str(images_path), mmap_mode='r'),
            'segmentation': np.load(str(segmentation_path), mmap_mode='r'),
        }
        if self.cache_max_cases > 0:
            self._case_cache[patient_id] = cached_case
            while len(self._case_cache) > self.cache_max_cases:
                self._case_cache.popitem(last=False)
        return cached_case

    def _random_negative_box(self, height, width, sample_index, class_index):
        rng = np.random.default_rng(self.sample_seed + sample_index * 31 + class_index)
        box_width = max(4, int(round(width * rng.uniform(0.10, 0.45))))
        box_height = max(4, int(round(height * rng.uniform(0.10, 0.45))))
        x_min = int(rng.integers(0, max(width - box_width + 1, 1)))
        y_min = int(rng.integers(0, max(height - box_height + 1, 1)))
        return torch.tensor([x_min, y_min, x_min + box_width - 1, y_min + box_height - 1], dtype=torch.float32)

    def _jitter_box(self, box, sample_index, class_index, img_size):
        """Apply PJT jitter to a positive oracle box. Deterministic per (sample, class)."""
        rng = np.random.default_rng(self.pjt_seed + sample_index * 31 + class_index)
        x_min, y_min, x_max, y_max = box.tolist()
        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1

        if box_w <= 0 or box_h <= 0:
            return box  # empty box, no jitter

        # Miss simulation
        if self.pjt_miss_prob > 0 and rng.random() < self.pjt_miss_prob:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)

        # Translation jitter
        translate_x = rng.uniform(-self.pjt_translate_max, self.pjt_translate_max) * box_w
        translate_y = rng.uniform(-self.pjt_translate_max, self.pjt_translate_max) * box_h

        # Scale jitter
        scale = rng.uniform(self.pjt_scale_min, self.pjt_scale_max)
        cx = (x_min + x_max) / 2.0 + translate_x
        cy = (y_min + y_max) / 2.0 + translate_y
        half_w = box_w * scale / 2.0
        half_h = box_h * scale / 2.0

        x_min_j = int(round(cx - half_w))
        y_min_j = int(round(cy - half_h))
        x_max_j = int(round(cx + half_w))
        y_max_j = int(round(cy + half_h))

        # Clip to image bounds with minimum extent
        x_min_j = max(0, min(x_min_j, img_size - 2))
        y_min_j = max(0, min(y_min_j, img_size - 2))
        x_max_j = max(x_min_j + 1, min(x_max_j, img_size - 1))
        y_max_j = max(y_min_j + 1, min(y_max_j, img_size - 1))

        return torch.tensor([x_min_j, y_min_j, x_max_j, y_max_j], dtype=torch.float32)

    def __getitem__(self, idx):
        patient_id, slice_idx = self.slice_list[idx]
        patient_folder = os.path.join(self.data_path, patient_id)

        if self.cache_root is not None:
            cached_case = self._load_cached_case(patient_id)
            image = np.asarray(cached_case['images'][:, slice_idx, :, :], dtype=np.float32)
            seg_slice = np.asarray(cached_case['segmentation'][slice_idx, :, :])
            image = np.stack(
                [
                    cv2.resize(channel, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                    for channel in image
                ],
                axis=0,
            )
        else:
            modalities = ['t1', 't1ce', 't2', 'flair']
            image_channels = []
            for mod in modalities:
                mod_path = os.path.join(patient_folder, f'{patient_id}_{mod}.nii.gz')
                mod_vol = sitk.GetArrayFromImage(sitk.ReadImage(mod_path))
                slice_2d = mod_vol[slice_idx, :, :]
                slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
                slice_2d = cv2.resize(slice_2d, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                image_channels.append(slice_2d)
            image = np.stack(image_channels, axis=0)

            seg_path = os.path.join(patient_folder, f'{patient_id}_seg.nii.gz')
            seg_vol = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
            seg_slice = seg_vol[slice_idx, :, :]
        seg_slice = cv2.resize(seg_slice, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        wt_mask = np.isin(seg_slice, [1, 2, 4]).astype(np.float32)
        tc_mask = np.isin(seg_slice, [1, 4]).astype(np.float32)
        et_mask = (seg_slice == 4).astype(np.float32)
        label = np.stack([et_mask, tc_mask, wt_mask], axis=0)  # Shape: (3, H, W)

        # --- 应用数据增强 ---
        if self.transform:
            # Albumentations 需要 (H, W, C) 格式的图像和 (H, W) 的掩码列表
            augmented = self.transform(image=image.transpose(1, 2, 0), masks=[label[0], label[1], label[2]])
            image = augmented['image'].transpose(2, 0, 1)  # 转回 (C, H, W)
            label = np.stack(augmented['masks'], axis=0)

        boxes = []
        for i in range(self.num_classes):
            box = get_main_bounding_box(torch.from_numpy(label[i]))
            box_is_empty = (box[2] - box[0] <= 0) or (box[3] - box[1] <= 0)
            if (
                self.negative_prompt_box == 'random'
                and not np.any(label)
            ):
                box = self._random_negative_box(self.image_size, self.image_size, idx, i)
            elif self.pjt_enabled and self.mode == 'train' and not box_is_empty:
                box = self._jitter_box(box, idx, i, self.image_size)
            boxes.append(box)
        boxes_tensor = torch.stack(boxes, dim=0)

        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).float(),
            "boxes": boxes_tensor,
        }
