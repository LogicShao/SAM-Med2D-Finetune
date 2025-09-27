import subprocess
import sys


def main():
    # 定义命令参数
    cmd = [
        sys.executable, "train_singletask.py",  # 使用当前 python 解释器
        "--finetune_method", "adapter",
        "--data_path", "data_brats_WT_TC_ET",
        "--work_dir", "workdir_label_WT_TC_ET",
        "--run_name", "single_task",
        "--model_type", "vit_b",
        "--sam_checkpoint", "pretrain_model/sam-med2d_b.pth",
        "--epochs", "200",
        "--batch_size", "10",
        "--image_size", "256",
        "--lr", "0.00001",
        "--device", "cuda",
        "--early_stopping_patience", "10",
        "--encoder_adapter", "True",
    ]

    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"运行出错，错误码：{e.returncode}")
        print(f"错误信息：{e}")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    main()
