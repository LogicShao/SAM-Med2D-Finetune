import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_env():
    env = os.environ.copy()
    src_root = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_root if not existing else os.pathsep.join((src_root, existing))
    return env


def main():
    cmd = [
        sys.executable,
        "-m",
        "sam_med2d_finetune.training.train_multitask",
        "--train_data_path",
        "data_brats_raw/train",
        "--val_data_path",
        "data_brats_raw/val",
        "--work_dir",
        "workdir_multi_task",
        "--finetune_method",
        "lora",
        "--run_name",
        "finetune_no_stop",
        "--model_type",
        "vit_b",
        "--sam_checkpoint",
        "pretrain_model/sam-med2d_b.pth",
        "--epochs",
        "200",
        "--batch_size",
        "14",
        "--image_size",
        "256",
        "--lr",
        "0.00001",
        "--device",
        "cuda",
        "--early_stopping_patience",
        "100",
        "--lora_r",
        "8",
        "--lora_alpha",
        "16",
        "--encoder_adapter",
        "True",
    ]
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=_build_env())
    except subprocess.CalledProcessError as e:
        print(f"运行出错，错误码：{e.returncode}")
        print(f"错误信息：{e}")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    main()
