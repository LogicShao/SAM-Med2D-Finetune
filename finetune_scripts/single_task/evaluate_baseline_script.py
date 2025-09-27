import subprocess
import sys


def main():
    cmd = [
        sys.executable, "evaluate_baseline.py",
        "--data_path", "data_brats_WT",
        "--work_dir", "workdir_label_WT_TC",
        "--image_size", "256",
        "--model_type", "vit_b",
        "--sam_checkpoint", "pretrain_model/sam-med2d_b.pth",
        "--device", "cuda",
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
