import argparse
import glob
import json
import logging
import os

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

# 配置日志，使其在终端提供清晰的输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_to_uint8(data):
    """将输入的Numpy数组归一化到0-255范围并转换为uint8类型。"""
    percentile_99 = np.percentile(data, 99)
    data = np.clip(data, 0, percentile_99)
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-9)
    return (data * 255).astype(np.uint8)


def preprocess_brats_data(raw_data_path, processed_data_path, selected_labels, mode, num_cases=None):
    """
    从 .nii.gz 文件转换BraTS数据以适配SAM-Med2D的DataLoader。

    Args:
        raw_data_path (str): 包含原始病例文件夹的路径。
        processed_data_path (str): 处理后数据的输出根目录。
        selected_labels (list): 要提取的标签列表 (e.g., ['WT', 'TC'])。
        mode (str): 当前处理的模式 ('train' 或 'val')。
        num_cases (int, optional): 要处理的病例数量上限。默认为 None (处理所有)。
    """
    logging.info(f"开始处理 '{mode}' 数据集，源路径: {raw_data_path}")
    logging.info(f"选择的标签: {selected_labels}")

    image_dir = os.path.join(processed_data_path, 'images', mode)
    label_dir = os.path.join(processed_data_path, 'labels', mode)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    image2label = {}
    label2image = {}

    case_paths = sorted(glob.glob(os.path.join(raw_data_path, f'BraTS2021_*')))
    if not case_paths:
        logging.warning(f"在路径 {raw_data_path} 中没有找到任何病例文件夹 (BraTS2021_*)。")
        return

    # 根据 num_cases 参数限制处理的病例数量
    if num_cases is not None and num_cases > 0:
        if num_cases < len(case_paths):
            logging.info(f"数据量限制已激活：将只处理前 {num_cases} 个病例（总共找到 {len(case_paths)} 个）。")
            case_paths = case_paths[:num_cases]
        else:
            logging.info(f"指定的病例数 ({num_cases}) 大于或等于找到的病例总数 ({len(case_paths)})。将处理所有病例。")

    for case_path in tqdm(case_paths, desc=f"正在处理 {mode} Cases"):
        case_id = os.path.basename(case_path)

        try:
            # 文件路径
            t1ce_path = os.path.join(case_path, f'{case_id}_t1ce.nii.gz')
            t2_path = os.path.join(case_path, f'{case_id}_t2.nii.gz')
            flair_path = os.path.join(case_path, f'{case_id}_flair.nii.gz')
            seg_path = os.path.join(case_path, f'{case_id}_seg.nii.gz')

            # 加载NIfTI文件
            img_t1ce = nib.load(t1ce_path).get_fdata()
            img_t2 = nib.load(t2_path).get_fdata()
            img_flair = nib.load(flair_path).get_fdata()
            seg_mask_3d = nib.load(seg_path).get_fdata().astype(np.uint8)

            for i in range(seg_mask_3d.shape[2]):
                seg_slice = seg_mask_3d[:, :, i]

                if np.sum(seg_slice) == 0:
                    continue

                label_map = {
                    'WT': ((seg_slice == 1) | (seg_slice == 2) | (seg_slice == 4)),
                    'TC': ((seg_slice == 1) | (seg_slice == 4)),
                    'ET': (seg_slice == 4)
                }

                label_paths_for_slice = []
                for label_name in selected_labels:
                    mask_data = label_map.get(label_name)
                    if mask_data is not None and np.sum(mask_data) > 0:
                        slice_base_name = f"{case_id}_slice_{i:03d}"
                        processed_label_path = os.path.join(label_dir, f"{slice_base_name}_{label_name}.png")
                        cv2.imwrite(processed_label_path, (mask_data * 255).astype(np.uint8))
                        label_paths_for_slice.append(processed_label_path)

                if label_paths_for_slice:
                    t1ce_slice = normalize_to_uint8(img_t1ce[:, :, i])
                    t2_slice = normalize_to_uint8(img_t2[:, :, i])
                    flair_slice = normalize_to_uint8(img_flair[:, :, i])
                    stacked_image = np.stack([t1ce_slice, t2_slice, flair_slice], axis=-1)

                    processed_image_path = os.path.join(image_dir, f"{slice_base_name}.png")
                    cv2.imwrite(processed_image_path, stacked_image)

                    image2label[processed_image_path] = label_paths_for_slice
                    for p in label_paths_for_slice:
                        label2image[p] = processed_image_path

        except FileNotFoundError:
            logging.warning(f"跳过 {case_id}，因为缺少一个或多个 .nii.gz 文件。")
            continue
        except Exception as e:
            logging.error(f"处理 {case_id} 时发生未知错误: {e}")

    # 保存 JSON 文件
    with open(os.path.join(processed_data_path, f'image2label_{mode}.json'), 'w') as f:
        json.dump(image2label, f, indent=4)
    with open(os.path.join(processed_data_path, f'label2image_{mode}.json'), 'w') as f:
        json.dump(label2image, f, indent=4)

    logging.info(f"'{mode}' 数据集处理完成！JSON 文件已保存到: {processed_data_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="预处理 BraTS .nii.gz 数据集以适配 SAM-Med2D 微调框架")

    # --- 路径参数 ---
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="包含原始训练病例文件夹的路径。")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="包含原始验证病例文件夹的路径。")
    parser.add_argument("--processed_data_path", type=str, default="data_brats_processed",
                        help="处理后数据的输出根目录。")

    # --- 数据集控制参数 ---
    parser.add_argument("--labels", nargs='+', default=['WT', 'TC', 'ET'], choices=['WT', 'TC', 'ET'],
                        help="选择要提取和保存的标签。例如: --labels WT TC")
    parser.add_argument("--num_train_cases", type=int, default=None,
                        help="限制处理的训练病例（病人）数量，用于快速测试。默认为处理所有。")
    parser.add_argument("--num_val_cases", type=int, default=None,
                        help="限制处理的验证病例（病人）数量。默认为处理所有。")

    args = parser.parse_args()

    # --- 主执行逻辑 ---
    # 分别处理训练集和验证集

    # 处理训练数据
    if not os.path.exists(args.train_data_path):
        logging.error(f"错误：找不到指定的训练数据路径: {args.train_data_path}")
    else:
        preprocess_brats_data(
            raw_data_path=args.train_data_path,
            processed_data_path=args.processed_data_path,
            selected_labels=args.labels,
            mode='train',
            num_cases=args.num_train_cases
        )

    # 处理验证数据
    if not os.path.exists(args.val_data_path):
        logging.error(f"错误：找不到指定的验证数据路径: {args.val_data_path}")
    else:
        preprocess_brats_data(
            raw_data_path=args.val_data_path,
            processed_data_path=args.processed_data_path,
            selected_labels=args.labels,
            mode='val',
            num_cases=args.num_val_cases
        )
