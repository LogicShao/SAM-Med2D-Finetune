import json
import os
import random

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm

# Mapping BraTS labels to human-readable names for filenames
LABEL_NAME_MAP = {
    1: 'tumor_core',  # Necrotic and Non-Enhancing Tumor Core
    2: 'edema',  # Peritumoral Edema
    4: 'enhancing_tumor'  # GD-enhancing Tumor
}


def create_sam_finetune_dataset(brats_root_dir, output_dir, image_size=256, num_patients=None, test_split_ratio=0.2):
    """
    Converts the BraTS 2021 dataset into the specific format required for fine-tuning SAM-Med2D.

    Args:
        brats_root_dir (str): Root directory of the BraTS 2021 dataset.
        output_dir (str): Directory where the formatted dataset will be saved.
        image_size (int, optional): The target size (height and width) for the output images and masks. Defaults to 256.
        num_patients (int, optional): Total number of patients to process. If None, all patients are used.
        test_split_ratio (float, optional): Proportion of patients to allocate to the test set.
    """
    print(f"--- Creating SAM-Med2D Fine-tuning Dataset in '{output_dir}' ---")
    print(
        f"Images and masks will be resized to {image_size}x{image_size} pixels.")

    # 1. Create directory structure
    images_path = os.path.join(output_dir, 'images')
    masks_path = os.path.join(output_dir, 'masks')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)

    # 2. Get and split patient IDs
    patient_ids = [d for d in os.listdir(brats_root_dir) if d.startswith(
        'BraTS') and os.path.isdir(os.path.join(brats_root_dir, d))]
    random.shuffle(patient_ids)

    if num_patients is not None and 0 < num_patients < len(patient_ids):
        print(
            f"Limiting processing to {num_patients} out of {len(patient_ids)} total patients.")
        patient_ids = patient_ids[:num_patients]

    split_index = int(len(patient_ids) * (1 - test_split_ratio))
    train_patients = patient_ids[:split_index]
    test_patients = patient_ids[split_index:]

    print(f"Total patients to process: {len(patient_ids)}")
    print(
        f"Splitting -> Train: {len(train_patients)}, Test: {len(test_patients)}")

    # 3. Initialize dictionaries for JSON files
    image2label_train = {}
    label2image_test = {}

    patient_lists = {'train': train_patients, 'test': test_patients}
    total_masks_created = 0
    total_images_created = 0

    target_size = (image_size, image_size)

    # 4. Process each patient
    for subset, patients in patient_lists.items():
        print(f"\nProcessing subset: {subset}")
        for patient_id in tqdm(patients, desc=f"Processing {subset} patients"):
            patient_dir = os.path.join(brats_root_dir, patient_id)

            flair_path = os.path.join(
                patient_dir, f"{patient_id}_flair.nii.gz")
            seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")

            if not (os.path.exists(flair_path) and os.path.exists(seg_path)):
                continue

            flair_data = nib.load(flair_path).get_fdata()
            seg_data = nib.load(seg_path).get_fdata()

            # 5. Process each slice
            for i in range(seg_data.shape[2]):
                seg_slice = seg_data[:, :, i]
                unique_labels = np.unique(seg_slice)

                if np.any(unique_labels > 0):

                    # 1. 加载原始 MRI 切片
                    mri_slice = flair_data[:, :, i]

                    # 2. 应用窗宽窗位（windowing）以增强对比度并处理异常值
                    pixels = mri_slice[mri_slice > 0]
                    if pixels.size > 0:
                        lower, upper = np.percentile(pixels, [1, 99])
                        mri_slice = np.clip(mri_slice, lower, upper)

                    # 3. 将经过窗宽窗位处理的切片线性缩放到 0-255 范围
                    #    注意：这是简单的线性映射，不是对整张图做 min-max 归一化
                    #    这样可以保留窗宽窗位后的相对灰度关系
                    if upper > lower:
                        slice_0_255 = ((mri_slice - lower) / (upper - lower) * 255.0).astype(np.uint8)
                    else:  # 处理全黑或单一灰度的情况
                        slice_0_255 = np.zeros_like(mri_slice, dtype=np.uint8)

                    # 4. 转换为三通道 BGR 图像
                    slice_rgb = cv2.cvtColor(slice_0_255, cv2.COLOR_GRAY2BGR)

                    # 5. 调整图像尺寸（此处缩放，而非依赖训练脚本中的 transform）
                    slice_resized = cv2.resize(slice_rgb, target_size, interpolation=cv2.INTER_AREA)

                    filename_base = f"{patient_id}_slice_{i:03d}"
                    image_filename = f"{filename_base}.png"

                    cv2.imwrite(os.path.join(
                        images_path, image_filename), slice_resized)
                    total_images_created += 1

                    relative_image_path = os.path.join(
                        os.path.basename(output_dir), 'images', image_filename)
                    current_slice_mask_paths = []

                    # --- Save individual masks for each tumor component ---
                    for label in unique_labels:
                        if label == 0:
                            continue

                        label_name = LABEL_NAME_MAP.get(
                            label, f"unknown_label_{int(label)}")
                        mask_filename = f"{filename_base}_{label_name}_000.png"
                        relative_mask_path = os.path.join(
                            os.path.basename(output_dir), 'masks', mask_filename)

                        binary_mask = (seg_slice == label).astype(
                            np.uint8) * 255

                        # --- NEW: Resize the mask using nearest-neighbor interpolation ---
                        mask_resized = cv2.resize(
                            binary_mask, target_size, interpolation=cv2.INTER_NEAREST)

                        cv2.imwrite(os.path.join(
                            masks_path, mask_filename), mask_resized)
                        total_masks_created += 1

                        current_slice_mask_paths.append(relative_mask_path)

                    if subset == 'train':
                        image2label_train[relative_image_path] = current_slice_mask_paths
                    else:
                        for mask_path in current_slice_mask_paths:
                            label2image_test[mask_path] = relative_image_path

    # 6. Save the JSON files
    with open(os.path.join(output_dir, 'image2label_train.json'), 'w') as f:
        json.dump(image2label_train, f, indent=4)

    with open(os.path.join(output_dir, 'label2image_test.json'), 'w') as f:
        json.dump(label2image_test, f, indent=4)

    print("\n--- SAM-Med2D Dataset Creation Finished! ---")
    print(f"Total images created: {total_images_created}")
    print(f"Total masks created: {total_masks_created}")
    print(f"Dataset saved to: {os.path.abspath(output_dir)}")
    print("JSON mapping files have been successfully generated.")


if __name__ == '__main__':
    # --- User Configuration ---

    # 1. Desired output size for images and masks (width and height)
    #    SAM-Med2D was pre-trained on 256x256 images.
    IMAGE_OUTPUT_SIZE = 256

    # 2. Number of patients to process
    #    - Set to None to use all available patients
    #    - Set to an integer (e.g., 100) for a smaller test dataset
    NUM_PATIENTS_TO_PROCESS = 10

    # 3. Train/Test split ratio
    #    - This value represents the proportion of patients reserved for the test set
    TEST_SPLIT_RATIO = 0.2  # 20% for testing, 80% for training

    # 4. Input and Output paths
    BRATS_ROOT_DIR = "./dataset_for_SAM-Med2D/train"
    OUTPUT_DATASET_DIR = "./dataset_for_SAM-Med2D_train"

    # --- Run the main function ---
    create_sam_finetune_dataset(
        brats_root_dir=BRATS_ROOT_DIR,
        output_dir=OUTPUT_DATASET_DIR,
        image_size=IMAGE_OUTPUT_SIZE,
        num_patients=NUM_PATIENTS_TO_PROCESS,
        test_split_ratio=TEST_SPLIT_RATIO
    )
