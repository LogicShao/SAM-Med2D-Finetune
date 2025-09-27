import subprocess
import sys


def main():
    cmd = [
        sys.executable, "preprocess_brats.py",
        "--train_data_path", "data_brats_raw/train",
        "--val_data_path", "data_brats_raw/val",
        "--processed_data_path", "data_brats_WT_TC",
        "--labels", "WT", "TC",  # "ET",
        # "--num_train_cases", "10",
        # "--num_val_cases", "2",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"运行出错，错误码：{e.returncode}")
        print(f"错误信息：{e}")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    main()
