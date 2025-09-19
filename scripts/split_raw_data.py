import os
import random
import shutil

from tqdm import tqdm


def split_brats_dataset(source_dir, dest_dir, test_split_ratio=0.2):
    """
    安全地将BraTS 2021数据集（按病人文件夹组织）划分为训练集和测试集。

    此函数通过复制文件来操作，原始数据集将保持不变。

    Args:
        source_dir (str): 原始BraTS训练数据的路径，其中包含BraTS2021_xxxxx格式的文件夹。
        dest_dir (str): 目标路径，将在此处创建'train'和'test'子文件夹。
        test_split_ratio (float): 用于测试集的数据比例，例如0.2表示20%的数据用于测试。
    """
    print("--- Starting BraTS Dataset Splitting Process ---")

    # --- 1. 查找所有有效的病人文件夹 ---
    print(f"Scanning source directory: {source_dir}")
    try:
        # 确保我们只获取目录，并排除任何可能存在的非目录文件
        patient_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        if not patient_dirs:
            print(f"错误：在源目录 '{source_dir}' 中未找到任何病人文件夹。请检查路径。")
            return
        print(f"Found {len(patient_dirs)} patients.")
    except FileNotFoundError:
        print(f"错误：源目录不存在于 '{source_dir}'。请检查路径。")
        return

    # --- 2. 随机打乱并划分病人列表 ---
    random.shuffle(patient_dirs)
    split_index = int(len(patient_dirs) * (1 - test_split_ratio))
    train_patients = patient_dirs[:split_index]
    test_patients = patient_dirs[split_index:]

    print(f"Splitting data into {len(train_patients)} training samples and {len(test_patients)} testing samples.")

    # --- 3. 创建目标目录 ---
    train_dest = os.path.join(dest_dir, 'train')
    test_dest = os.path.join(dest_dir, 'test')
    os.makedirs(train_dest, exist_ok=True)
    os.makedirs(test_dest, exist_ok=True)
    print(f"Created destination directories:\n  - {train_dest}\n  - {test_dest}")

    # --- 4. 定义一个辅助函数来复制文件并显示进度条 ---
    def copy_patient_data(patient_list, destination_folder, description):
        """复制指定列表中的所有病人文件夹到目标位置。"""
        print(f"\nCopying {description} data...")
        for patient in tqdm(patient_list, desc=description):
            source_path = os.path.join(source_dir, patient)
            dest_path = os.path.join(destination_folder, patient)

            # 如果目标文件夹已存在，先删除，以防万一（例如，中断后重新运行）
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)

            shutil.copytree(source_path, dest_path)

    # --- 5. 执行复制操作 ---
    copy_patient_data(train_patients, train_dest, "Training")
    copy_patient_data(test_patients, test_dest, "Testing")

    print("\n--- Dataset Splitting Complete! ---")
    print(f"Data is now located in: {dest_dir}")


if __name__ == "__main__":
    # --- 用户配置区 ---

    # 1. 输入：您的原始BraTS数据集所在的路径
    #    使用 r"..." 语法可以防止Windows路径中的反斜杠 `\` 出现问题
    SOURCE_DATA_DIR = r"D:\proj\Brain-Tumor-Segmentation\BraTs2021\BraTS2021_Training_Data"

    # 2. 输出：您希望存放划分后数据集的根目录
    DESTINATION_DIR = r"D:\proj\SAM-Med2D\dataset_for_SAM-Med2D"

    # 3. 比例：测试集所占的比例
    TEST_RATIO = 0.2  # 20% 的数据将用于测试, 80% 用于训练

    # --- 执行脚本 ---
    split_brats_dataset(SOURCE_DATA_DIR, DESTINATION_DIR, TEST_RATIO)
