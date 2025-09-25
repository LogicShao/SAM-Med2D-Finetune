import subprocess
import sys


def main():
    cmd = [
        sys.executable, "evaluate_baseline.py",
        "--data_path", "data_brats_processed",
        "--work_dir", "workdir_label_WT_TC_ET",
        "--image_size", "256",
        "--model_type", "vit_b",
        "--sam_checkpoint", "pretrain_model/sam-med2d_b.pth",
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
