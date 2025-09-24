import os

import SimpleITK as sitk
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_main_bounding_box(mask_tensor):
    y_indices, x_indices = torch.where(mask_tensor > 0)
    if len(y_indices) == 0:
        return torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    x_min, x_max = torch.min(x_indices), torch.max(x_indices)
    y_min, y_max = torch.min(y_indices), torch.max(y_indices)
    return torch.tensor([x_min.item(), y_min.item(), x_max.item(), y_max.item()], dtype=torch.float32)


class BraTSDataset(Dataset):
    def __init__(self, data_path, image_size=256, num_classes=3, mode='train', subset_size=None):
        self.data_path = data_path
        self.image_size = image_size
        self.num_classes = num_classes
        self.mode = mode

        self.patients = sorted([p for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))])

        self.slice_list = []
        # print(f"Preprocessing BraTS {self.mode} dataset to find valid slices...")
        for patient_id in self.patients:
            patient_folder = os.path.join(self.data_path, patient_id)
            seg_path = os.path.join(patient_folder, f'{patient_id}_seg.nii.gz')
            if os.path.exists(seg_path):
                seg_vol = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
                for slice_idx in range(seg_vol.shape[0]):
                    if np.sum(seg_vol[slice_idx, :, :]) > 0:
                        self.slice_list.append((patient_id, slice_idx))

        if subset_size is not None:
            self.slice_list = self.slice_list[:subset_size]
        print(f"Found {len(self.slice_list)} valid slices for {self.mode} set.")

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
                    border_mode=cv2.BORDER_CONSTANT
                ),
                A.RandomBrightnessContrast(p=0.5),
                A.ElasticTransform(p=0.5, border_mode=cv2.BORDER_CONSTANT),
            ])
        else:
            # 验证集通常不需要数据增强
            self.transform = None

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, idx):
        patient_id, slice_idx = self.slice_list[idx]
        patient_folder = os.path.join(self.data_path, patient_id)

        modalities = ['t1', 't1ce', 't2', 'flair']
        image_channels = []
        for mod in modalities:
            mod_path = os.path.join(patient_folder, f'{patient_id}_{mod}.nii.gz')
            mod_vol = sitk.GetArrayFromImage(sitk.ReadImage(mod_path))
            slice_2d = mod_vol[slice_idx, :, :]
            slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
            slice_2d = cv2.resize(slice_2d, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            image_channels.append(slice_2d)
        image = np.stack(image_channels, axis=0)  # Shape: (4, H, W)

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
            boxes.append(box)
        boxes_tensor = torch.stack(boxes, dim=0)

        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.from_numpy(label).float(),
            "boxes": boxes_tensor,
        }
