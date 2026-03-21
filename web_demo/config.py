from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DEMO_ROOT = PROJECT_ROOT / "web_demo"
STATIC_DIR = WEB_DEMO_ROOT / "static"
TEMPLATES_DIR = WEB_DEMO_ROOT / "templates"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEMO_RUNS_DIR = OUTPUTS_DIR / "web_demo_runs"
UPLOAD_STAGE_DIR = DEMO_RUNS_DIR / "uploads"
GENERATED_IMAGE_DIR = STATIC_DIR / "generated"

APP_NAME = "SAM-Med2D 脑肿瘤最小演示版"
APP_DESCRIPTION = "只保留单条主链路：上传或选择病例，自动分割，3D 重建，结果查看。"
APP_HOST = os.getenv("WEB_DEMO_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("WEB_DEMO_PORT", "7860"))
PYTHON_EXECUTABLE = Path(os.getenv("WEB_DEMO_PYTHON", sys.executable)).resolve()

SAMPLE_CASE_COLLECTIONS = (
    {
        "label": "现成演示样例",
        "tag": "四病例闭环",
        "root": OUTPUTS_DIR / "postprocess_yolo_box_4cases",
        "max_cases": 4,
    },
    {
        "label": "困难病例样例",
        "tag": "困难病例",
        "root": OUTPUTS_DIR / "stage2_hard8_top1",
        "max_cases": 2,
    },
)

VIEWER_CANDIDATES = (
    "viewer.html",
    "preview_3d_compare_all.html",
    "preview_3d_all.html",
    "preview_3d_compare_combined.html",
    "preview_3d_combined.html",
)
SUMMARY_JSON_CANDIDATES = ("summary.json", "summary_metrics.json", "metrics.json")
SUMMARY_MD_CANDIDATES = ("summary.md",)
METRIC_CANDIDATES = ("summary_metrics.json", "metrics.json", "postprocess_report.json", "prompt_stats.json")

RAW_MASK_FILES = ("ET.nii.gz", "TC.nii.gz", "WT.nii.gz", "combined_label.nii.gz")
POST_MASK_FILES = ("post_ET.nii.gz", "post_TC.nii.gz", "post_WT.nii.gz", "post_combined_label.nii.gz")
PIPELINE_SUMMARY_FILE = "summary.json"

DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "pretrain_model" / "sam-med2d_b.pth"
DEFAULT_FINETUNED_CHECKPOINT = (
    PROJECT_ROOT / "workdir_multi_task" / "models" / "finetune_no_stop_lora" / "lora_adapters"
)
DEFAULT_YOLO_CHECKPOINT = PROJECT_ROOT / "workdir_yolo" / "brats_yolo_dev_img320_v8m" / "weights" / "best.pt"


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEFAULT_INFERENCE_ARGS = {
    "sam_checkpoint": DEFAULT_SAM_CHECKPOINT,
    "finetuned_checkpoint": DEFAULT_FINETUNED_CHECKPOINT,
    "finetune_method": "lora",
    "prompt_mode": "yolo_box",
    "model_type": "vit_b",
    "image_size": 256,
    "input_channels": 4,
    "encoder_adapter": True,
    "threshold": 0.5,
    "use_amp": True,
    "device": default_device(),
    "yolo_checkpoint": DEFAULT_YOLO_CHECKPOINT,
    "yolo_conf": 0.05,
    "yolo_iou": 0.60,
    "yolo_max_det": 2,
    "yolo_topk": 2,
    "prompt_box_strategy": "top1",
    "z_prompt_mode": "none",
    "wt_continuity_enabled": False,
}

DEFAULT_POSTPROCESS_ARGS = {
    "closing_radius": 2,
    "opening_radius": 1,
    "wt_keep_largest": True,
    "keep_topk_tc": 1,
    "keep_topk_et": 1,
    "z_smooth_iterations": 3,
}


def ensure_web_demo_dirs() -> None:
    for path in (STATIC_DIR, TEMPLATES_DIR, DEMO_RUNS_DIR, UPLOAD_STAGE_DIR, GENERATED_IMAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)
