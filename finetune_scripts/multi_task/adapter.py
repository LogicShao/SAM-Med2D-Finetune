import subprocess
import sys


def main():
    cmd = [
        sys.executable, "train_multitask.py",
        "--train_data_path", "data_brats_raw/train",
        "--val_data_path", "data_brats_raw/val",
        "--work_dir", "workdir_multi_task",
        "--finetune_method", "adapter",
        "--run_name", "finetune",
        "--model_type", "vit_b",
        "--sam_checkpoint", "pretrain_model/sam-med2d_b.pth",
        "--epochs", "200",
        "--batch_size", "10",
        "--image_size", "256",
        "--lr", "0.00001",
        "--device", "cuda",
        "--early_stopping_patience", "10",
        "--encoder_adapter", "True",
        # 如有 adapter 特有参数可在此添加
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
