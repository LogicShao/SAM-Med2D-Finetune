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

APP_NAME = "脑肿瘤 MRI 三维可视化辅助分析平台"
APP_DESCRIPTION = "支持稳健默认配置与多类别分割分析两种模式。"
APP_HOST = os.getenv("WEB_DEMO_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("WEB_DEMO_PORT", "7860"))
PYTHON_EXECUTABLE = Path(os.getenv("WEB_DEMO_PYTHON", sys.executable)).resolve()

SAMPLE_CASE_COLLECTIONS = (
    {
        "label": "病例结果",
        "tag": "常规病例",
        "root": OUTPUTS_DIR / "postprocess_yolo_box_4cases",
        "max_cases": 4,
    },
    {
        "label": "病例结果",
        "tag": "复杂病例",
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
SAMPLE_RESULT_COLLECTIONS = {
    "standard": (
        {
            "label": "病例结果",
            "tag": "标准模式",
            "root": OUTPUTS_DIR / "stage7_adapter_verification" / "fixed20_adapter_baseline",
            "max_cases": 6,
        },
    ),
    "multiclass": (
        {
            "label": "病例结果",
            "tag": "多类别分析",
            "root": OUTPUTS_DIR / "stage9_et_prompt_tuning" / "fixed20_adapter_class_boxes_points_et_default",
            "max_cases": 6,
        },
    ),
}
SUMMARY_JSON_CANDIDATES = ("summary.json", "summary_metrics.json", "metrics.json")
SUMMARY_MD_CANDIDATES = ("summary.md",)
METRIC_CANDIDATES = ("summary_metrics.json", "metrics.json", "postprocess_report.json", "prompt_stats.json")

RAW_MASK_FILES = ("ET.nii.gz", "TC.nii.gz", "WT.nii.gz", "combined_label.nii.gz")
POST_MASK_FILES = ("post_ET.nii.gz", "post_TC.nii.gz", "post_WT.nii.gz", "post_combined_label.nii.gz")
PIPELINE_SUMMARY_FILE = "summary.json"

DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "pretrain_model" / "sam-med2d_b.pth"
DEFAULT_FINETUNED_CHECKPOINT = (
    PROJECT_ROOT / "workdir_multi_task" / "models" / "finetune_adapter" / "best_model.pth"
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
    "finetune_method": "adapter",
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

DEFAULT_DEMO_MODE = "standard"

DEMO_MODES = {
    "standard": {
        "key": "standard",
        "label": "标准模式",
        "short_label": "稳健默认配置",
        "tagline": "重点查看整体病灶范围、三维结果与关键切片。",
        "description": "使用稳健默认配置，页面重点展示整体病灶范围、三维结果与结果查看。",
        "result_note": "当前页面展示稳健默认配置下的整体病灶分析结果。",
        "analysis_title": "整体病灶分析",
        "analysis_description": "当前页面重点展示整体病灶范围与体积信息，可用于辅助查看病灶总体分布。",
        "summary_title": "结果说明",
        "summary_description": "当前页面重点展示整体病灶范围、三维结果与关键切片。",
        "slice_title": "关键切片",
        "slice_description": "页面重点展示整体病灶范围与关键切片。",
        "viewer_title": "三维可视化结果",
        "viewer_description": "当前页面重点展示整体病灶范围的三维结果。",
        "viewer_mask_name": "WT",
        "analysis_card_keys": ("total", "wt"),
        "show_multiclass_details": False,
        "warning_copy": "若存在分区结果，可作为辅助参考，但不作为当前页面的主要展示重点。",
        "inference_overrides": {
            "class_prompt_variant": "baseline",
            "wt_continuity_enabled": False,
        },
        "postprocess_overrides": {},
    },
    "multiclass": {
        "key": "multiclass",
        "label": "多类别分析模式",
        "short_label": "多类别分割分析",
        "tagline": "重点查看 WT / TC / ET 区域分布、分类体积与三维结果。",
        "description": "使用当前默认的多类别 prompt 分析方案，页面重点展示不同肿瘤区域分布。",
        "result_note": "当前页面展示多类别分割分析结果，可用于辅助观察不同肿瘤区域分布。",
        "analysis_title": "多类别定量分析",
        "analysis_description": "当前页面展示 WT / TC / ET 分区结果，可用于辅助观察不同肿瘤区域分布。",
        "summary_title": "分析说明",
        "summary_description": "当前页面展示多类别分割分析结果，用于辅助观察不同肿瘤区域分布。",
        "slice_title": "分类切片 / 分割叠加",
        "slice_description": "页面展示多类别分割叠加结果，可用于辅助观察不同肿瘤区域分布。",
        "viewer_title": "分类三维结果",
        "viewer_description": "当前页面重点展示 WT / TC / ET 的分类三维结果。",
        "viewer_mask_name": "all",
        "analysis_card_keys": ("total", "wt", "tc", "et"),
        "show_multiclass_details": True,
        "warning_copy": "当前结果用于多类别辅助分析，不替代稳健默认配置的整体病灶查看。",
        "inference_overrides": {
            "class_prompt_variant": "class_boxes_points",
            "et_prompt_variant": "default",
            "wt_continuity_enabled": False,
        },
        "postprocess_overrides": {},
    },
}


def normalize_demo_mode(mode_key: str | None) -> str:
    mode_key = str(mode_key or DEFAULT_DEMO_MODE).strip().lower()
    if mode_key not in DEMO_MODES:
        return DEFAULT_DEMO_MODE
    return mode_key


def get_demo_mode(mode_key: str | None) -> dict[str, object]:
    normalized = normalize_demo_mode(mode_key)
    return {**DEMO_MODES[normalized]}


def list_demo_modes() -> list[dict[str, object]]:
    return [get_demo_mode(mode_key) for mode_key in DEMO_MODES]


def get_sample_result_collections(mode_key: str | None) -> tuple[dict[str, object], ...]:
    normalized = normalize_demo_mode(mode_key)
    return tuple(dict(item) for item in SAMPLE_RESULT_COLLECTIONS.get(normalized, ()))


def resolve_sample_result_mode(result_dir: Path) -> str | None:
    target = Path(result_dir).resolve()
    for mode_key, collections in SAMPLE_RESULT_COLLECTIONS.items():
        for collection in collections:
            root = Path(collection["root"]).resolve()
            if target == root or root in target.parents:
                return str(mode_key)
    return None


def ensure_web_demo_dirs() -> None:
    for path in (STATIC_DIR, TEMPLATES_DIR, DEMO_RUNS_DIR, UPLOAD_STAGE_DIR, GENERATED_IMAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)
