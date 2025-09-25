import subprocess
import sys


def main():
    cmd = [
        sys.executable, "preprocess_brats.py",
        "--train_data_path", "data_brats_raw/train",
        "--val_data_path", "data_brats_raw/val",
        "--processed_data_path", "data_brats_processed",
        "--labels", "WT", "TC", "ET",
        "--num_train_cases", "10",
        "--num_val_cases", "2",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
