import subprocess
import sys


def main():
    # 定义命令参数
    cmd = [
        sys.executable, "train_unified.py",  # 使用当前 python 解释器
        "--finetune_method", "adapter",
        "--data_path", "data_brats_processed",
        "--work_dir", "workdir_label_WT_TC_ET",
        "--run_name", "finetune",
        "--model_type", "vit_b",
        "--sam_checkpoint", "pretrain_model/sam-med2d_b.pth",
        "--epochs", "5",
        "--batch_size", "32",
        "--image_size", "256",
        "--lr", "0.00001",
        "--device", "cuda",
        "--early_stopping_patience", "10",
        "--encoder_adapter", "True",
    ]

    # 执行命令
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
