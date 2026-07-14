"""Configuration contracts shared by BraTS inference entry points."""

from pathlib import Path

from sam_med2d_finetune.brats.constants import BRATS_CLASS_NAMES


DEFAULT_YOLO_CHECKPOINT = "workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt"
DEFAULT_YOLO_IMGSZ = 320
PROMPT_BOX_STRATEGIES = ("top1", "top2_merge")
Z_PROMPT_MODES = ("none", "smooth", "interpolate")


def normalize_class_prompt_strategies(
    prompt_box_strategy,
    prompt_box_strategy_et=None,
    prompt_box_strategy_tc=None,
    prompt_box_strategy_wt=None,
):
    class_overrides = {
        "ET": prompt_box_strategy_et,
        "TC": prompt_box_strategy_tc,
        "WT": prompt_box_strategy_wt,
    }
    normalized = {}
    for class_name in BRATS_CLASS_NAMES:
        strategy = str(class_overrides[class_name] or prompt_box_strategy)
        if strategy not in PROMPT_BOX_STRATEGIES:
            raise ValueError(
                f"Unsupported prompt_box_strategy for {class_name}: {strategy}. "
                f"Expected one of {PROMPT_BOX_STRATEGIES}."
            )
        normalized[class_name] = strategy
    return normalized


def build_postprocess_config(
    enabled,
    closing_radius,
    opening_radius,
    wt_keep_largest,
    keep_topk_tc,
    keep_topk_et,
    z_smooth_iterations,
):
    return {
        "enabled": bool(enabled),
        "closing_radius": int(closing_radius),
        "opening_radius": int(opening_radius),
        "wt_keep_largest": bool(wt_keep_largest),
        "keep_topk_tc": int(keep_topk_tc),
        "keep_topk_et": int(keep_topk_et),
        "z_smooth_iterations": int(z_smooth_iterations),
    }


def build_yolo_prompt_config(
    prompt_mode,
    yolo_checkpoint,
    yolo_conf,
    yolo_iou,
    yolo_max_det,
    yolo_topk,
    prompt_box_strategy,
    prompt_box_strategy_et,
    prompt_box_strategy_tc,
    prompt_box_strategy_wt,
    top2_score_ratio,
    top2_area_ratio_min,
    top2_area_ratio_max,
    top2_iou_max,
    z_prompt_mode,
    z_smooth_window,
    z_fill_gap_max,
    z_center_shift_max,
    z_area_ratio_min,
    z_area_ratio_max,
    wt_continuity_enabled,
    wt_continuity_score_thresh,
    wt_continuity_center_shift_max,
    wt_continuity_area_ratio_min,
    wt_continuity_area_ratio_max,
    wt_continuity_mask_dilate_iters,
    wt_continuity_mask_blur_kernel,
    class_prompt_variant,
    et_prompt_variant,
):
    if prompt_mode != "yolo_box":
        return None
    class_strategies = normalize_class_prompt_strategies(
        prompt_box_strategy=prompt_box_strategy,
        prompt_box_strategy_et=prompt_box_strategy_et,
        prompt_box_strategy_tc=prompt_box_strategy_tc,
        prompt_box_strategy_wt=prompt_box_strategy_wt,
    )
    return {
        "checkpoint": str(Path(yolo_checkpoint).resolve()),
        "conf": float(yolo_conf),
        "iou": float(yolo_iou),
        "imgsz": int(DEFAULT_YOLO_IMGSZ),
        "max_det": int(yolo_max_det),
        "topk": int(yolo_topk),
        "box_strategy": str(prompt_box_strategy),
        "box_strategy_by_class": class_strategies,
        "top2_rules": {
            "score_ratio": float(top2_score_ratio),
            "area_ratio_min": float(top2_area_ratio_min),
            "area_ratio_max": float(top2_area_ratio_max),
            "box_iou_max": float(top2_iou_max),
        },
        "z_prompt": {
            "mode": str(z_prompt_mode),
            "smooth_window": int(z_smooth_window),
            "fill_gap_max": int(z_fill_gap_max),
            "center_shift_max": float(z_center_shift_max),
            "area_ratio_min": float(z_area_ratio_min),
            "area_ratio_max": float(z_area_ratio_max),
        },
        "wt_continuity": {
            "enabled": bool(wt_continuity_enabled),
            "score_thresh": float(wt_continuity_score_thresh),
            "center_shift_max": float(wt_continuity_center_shift_max),
            "area_ratio_min": float(wt_continuity_area_ratio_min),
            "area_ratio_max": float(wt_continuity_area_ratio_max),
            "mask_dilate_iters": int(wt_continuity_mask_dilate_iters),
            "mask_blur_kernel": int(wt_continuity_mask_blur_kernel),
        },
        "class_prompt_variant": str(class_prompt_variant),
        "et_prompt_variant": str(et_prompt_variant),
    }
