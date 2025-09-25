import subprocess
import sys


def main():
    cmd = [
        sys.executable, "train_unified.py",
        "--finetune_method", "lora",
        "--data_path", "data_brats_processed",
        "--work_dir", "workdir_label_WT_TC_ET",
        "--run_name", "finetune",
        "--model_type", "vit_b",
        "--sam_checkpoint", "pretrain_model/sam-med2d_b.pth",
        "--epochs", "5",
        "--batch_size", "16",
        "--image_size", "256",
        "--lr", "0.00001",
        "--device", "cuda",
        "--early_stopping_patience", "10",
        "--lora_r", "8",
        "--lora_alpha", "16",
        "--encoder_adapter", "True",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
