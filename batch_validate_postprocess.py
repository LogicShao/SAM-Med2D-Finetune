import argparse
import csv
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brats_metrics import evaluate_brats_case, mean_defined
from brats_case import BraTSCase
from brats_constants import BRATS_CLASS_NAMES as CLASS_NAMES
from cli_utils import resolve_torch_device, str_to_bool
from inference_config import (
    DEFAULT_YOLO_CHECKPOINT,
    PROMPT_BOX_STRATEGIES,
    build_postprocess_config,
    build_yolo_prompt_config,
)
from inference_io import (
    build_case_meta,
    build_combined_label,
    save_case_meta,
    save_json,
    save_mask_outputs,
)
from infer_volume import (
    build_prompt_provider,
    run_volume_inference,
)
from model_factory import load_multitask_model
from postprocess_3d import postprocess_brats_masks
from prompt_strategies import (
    CLASS_PROMPT_VARIANTS,
    ET_PROMPT_VARIANTS,
    analyze_class_volume_consistency,
    merge_prompt_source_counts,
    summarize_consistency_across_cases,
)


LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch validation for raw vs post-processed BraTS volume inference.")
    parser.add_argument("--cases_root", required=True, help="Directory containing BraTS case folders.")
    parser.add_argument("--output_root", required=True, help="Directory to store per-case outputs and summary files.")
    parser.add_argument("--case_ids", nargs="+", default=None, help="Optional explicit case IDs to process.")
    parser.add_argument("--max_cases", type=int, default=4, help="Max number of cases when --case_ids is not set.")
    parser.add_argument("--sam_checkpoint", required=True, help="Base SAM-Med2D checkpoint path.")
    parser.add_argument("--finetuned_checkpoint", required=True, help="Adapter .pth or LoRA adapter directory.")
    parser.add_argument("--finetune_method", required=True, choices=["adapter", "lora"])
    parser.add_argument("--prompt_mode", default="full_image_box")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--input_channels", type=int, default=4)
    parser.add_argument("--encoder_adapter", type=str_to_bool, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_amp", type=str_to_bool, default=True)
    parser.add_argument("--disable_cudnn", type=str_to_bool, default=False)
    parser.add_argument("--yolo_checkpoint", default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--yolo_conf", type=float, default=0.05)
    parser.add_argument("--yolo_iou", type=float, default=0.60)
    parser.add_argument("--yolo_max_det", type=int, default=2)
    parser.add_argument("--yolo_topk", type=int, default=2)
    parser.add_argument("--prompt_box_strategy", default="top1", choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--prompt_box_strategy_et", default=None, choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--prompt_box_strategy_tc", default=None, choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--prompt_box_strategy_wt", default=None, choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--class_prompt_variant", default="baseline", choices=CLASS_PROMPT_VARIANTS)
    parser.add_argument("--et_prompt_variant", default="default", choices=ET_PROMPT_VARIANTS)
    parser.add_argument("--top2_score_ratio", type=float, default=0.5)
    parser.add_argument("--top2_area_ratio_min", type=float, default=0.1)
    parser.add_argument("--top2_area_ratio_max", type=float, default=2.0)
    parser.add_argument("--top2_iou_max", type=float, default=0.9)
    parser.add_argument("--z_prompt_mode", default="none", choices=["none", "smooth", "interpolate"])
    parser.add_argument("--z_smooth_window", type=int, default=1)
    parser.add_argument("--z_fill_gap_max", type=int, default=1)
    parser.add_argument("--z_center_shift_max", type=float, default=64.0)
    parser.add_argument("--z_area_ratio_min", type=float, default=0.25)
    parser.add_argument("--z_area_ratio_max", type=float, default=4.0)
    parser.add_argument("--wt_continuity_enabled", type=str_to_bool, default=False)
    parser.add_argument("--wt_continuity_score_thresh", type=float, default=0.15)
    parser.add_argument("--wt_continuity_center_shift_max", type=float, default=48.0)
    parser.add_argument("--wt_continuity_area_ratio_min", type=float, default=0.5)
    parser.add_argument("--wt_continuity_area_ratio_max", type=float, default=2.0)
    parser.add_argument("--wt_continuity_mask_dilate_iters", type=int, default=1)
    parser.add_argument("--wt_continuity_mask_blur_kernel", type=int, default=3)
    parser.add_argument("--postprocess", type=str_to_bool, default=True)
    parser.add_argument("--closing_radius", type=int, default=1)
    parser.add_argument("--opening_radius", type=int, default=1)
    parser.add_argument("--wt_keep_largest", type=str_to_bool, default=True)
    parser.add_argument("--keep_topk_tc", type=int, default=2)
    parser.add_argument("--keep_topk_et", type=int, default=2)
    parser.add_argument("--z_smooth_iterations", type=int, default=1)
    parser.add_argument(
        "--html_mask_name",
        default="all",
        choices=["all", "ET", "TC", "WT", "combined"],
        help="Mask configuration used for the preview HTML.",
    )
    parser.add_argument("--html_opacity", type=float, default=0.45)
    parser.add_argument("--render_html", type=str_to_bool, default=True)
    return parser.parse_args()


def select_case_dirs(cases_root, case_ids=None, max_cases=3):
    cases_root = Path(cases_root)
    if not cases_root.is_dir():
        raise FileNotFoundError(f"Cases root not found: {cases_root}")

    if case_ids:
        case_dirs = [cases_root / case_id for case_id in case_ids]
    else:
        all_cases = sorted(path for path in cases_root.iterdir() if path.is_dir())
        case_dirs = all_cases[: max(int(max_cases), 0)]

    missing = [str(path) for path in case_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Case directories not found: {missing}")
    return case_dirs


def load_ground_truth_masks(case_dir):
    case_dir = Path(case_dir)
    case_id = case_dir.name
    seg_path = case_dir / f"{case_id}_seg.nii.gz"
    if not seg_path.is_file():
        raise FileNotFoundError(f"Ground-truth segmentation not found: {seg_path}")

    segmentation_image = nib.load(str(seg_path))
    seg_volume = np.asarray(segmentation_image.dataobj, dtype=np.int16)
    masks = {
        "ET": (seg_volume == 4).astype(np.uint8),
        "TC": np.isin(seg_volume, [1, 4]).astype(np.uint8),
        "WT": np.isin(seg_volume, [1, 2, 4]).astype(np.uint8),
    }
    return masks, tuple(float(value) for value in segmentation_image.header.get_zooms()[:3])


def summarize_results(results):
    if not results:
        return {"num_cases": 0}

    summary = {
        "num_cases": len(results),
        "raw": {"mean_dice": 0.0, "mean_iou": 0.0, "per_class": {}, "hierarchy": {}},
        "post": {"mean_dice": 0.0, "mean_iou": 0.0, "per_class": {}, "hierarchy": {}},
        "delta": {"mean_dice": 0.0, "mean_iou": 0.0, "per_class": {}},
    }

    for mode in ("raw", "post"):
        summary[mode]["mean_dice"] = float(np.mean([item[mode]["mean_dice"] for item in results]))
        summary[mode]["mean_iou"] = float(np.mean([item[mode]["mean_iou"] for item in results]))
        for class_name in CLASS_NAMES:
            class_metrics = {
                metric_name: mean_defined(
                    [item[mode]["per_class"][class_name][metric_name] for item in results]
                )
                for metric_name in ("dice", "iou", "hd95_mm", "sensitivity", "specificity")
            }
            class_metrics["empty_ground_truth_cases"] = sum(
                item[mode]["per_class"][class_name]["empty_region"] in {"both_empty", "ground_truth_empty"}
                for item in results
            )
            summary[mode]["per_class"][class_name] = class_metrics
        summary[mode]["hierarchy"] = {
            "et_outside_tc_voxels": sum(item[mode]["hierarchy"]["et_outside_tc_voxels"] for item in results),
            "tc_outside_wt_voxels": sum(item[mode]["hierarchy"]["tc_outside_wt_voxels"] for item in results),
            "any_violation_voxels": sum(item[mode]["hierarchy"]["any_violation_voxels"] for item in results),
            "violation_case_count": sum(item[mode]["hierarchy"]["has_violation"] for item in results),
        }
        summary[mode]["hierarchy"]["violation_case_rate"] = (
            summary[mode]["hierarchy"]["violation_case_count"] / len(results)
        )

    summary["delta"]["mean_dice"] = summary["post"]["mean_dice"] - summary["raw"]["mean_dice"]
    summary["delta"]["mean_iou"] = summary["post"]["mean_iou"] - summary["raw"]["mean_iou"]
    for class_name in CLASS_NAMES:
        summary["delta"]["per_class"][class_name] = {
            "dice": summary["post"]["per_class"][class_name]["dice"] - summary["raw"]["per_class"][class_name]["dice"],
            "iou": summary["post"]["per_class"][class_name]["iou"] - summary["raw"]["per_class"][class_name]["iou"],
        }
    return summary


def summarize_wt_continuity(results):
    summary = {
        "enabled": False,
        "num_cases_with_stats": 0,
        "eligible_total": 0,
        "trigger_total": 0,
        "trigger_reasons": {},
        "rescue": 0,
        "neutral": 0,
        "harm": 0,
    }
    for item in results:
        wt_continuity = item.get("wt_continuity")
        if not wt_continuity:
            continue
        case_summary = wt_continuity.get("summary") or {}
        summary["num_cases_with_stats"] += 1
        summary["enabled"] = summary["enabled"] or bool(case_summary.get("enabled"))
        summary["eligible_total"] += int(case_summary.get("eligible_total", 0))
        summary["trigger_total"] += int(case_summary.get("trigger_total", 0))
        summary["rescue"] += int(case_summary.get("rescue", 0))
        summary["neutral"] += int(case_summary.get("neutral", 0))
        summary["harm"] += int(case_summary.get("harm", 0))
        for reason, count in (case_summary.get("trigger_reasons") or {}).items():
            summary["trigger_reasons"][str(reason)] = summary["trigger_reasons"].get(str(reason), 0) + int(count)
    return summary


def build_summary_row(case_result):
    row = {
        "case_id": case_result["case_id"],
        "output_dir": case_result["output_dir"],
        "html_path": case_result["html_path"] or "",
        "raw_mean_dice": case_result["raw"]["mean_dice"],
        "raw_mean_iou": case_result["raw"]["mean_iou"],
        "post_mean_dice": case_result["post"]["mean_dice"],
        "post_mean_iou": case_result["post"]["mean_iou"],
        "delta_mean_dice": case_result["post"]["mean_dice"] - case_result["raw"]["mean_dice"],
        "delta_mean_iou": case_result["post"]["mean_iou"] - case_result["raw"]["mean_iou"],
    }
    for class_name in CLASS_NAMES:
        row[f"raw_dice_{class_name}"] = case_result["raw"]["per_class"][class_name]["dice"]
        row[f"raw_iou_{class_name}"] = case_result["raw"]["per_class"][class_name]["iou"]
        row[f"post_dice_{class_name}"] = case_result["post"]["per_class"][class_name]["dice"]
        row[f"post_iou_{class_name}"] = case_result["post"]["per_class"][class_name]["iou"]
        row[f"delta_dice_{class_name}"] = (
            case_result["post"]["per_class"][class_name]["dice"] - case_result["raw"]["per_class"][class_name]["dice"]
        )
        row[f"delta_iou_{class_name}"] = (
            case_result["post"]["per_class"][class_name]["iou"] - case_result["raw"]["per_class"][class_name]["iou"]
        )
        for mode in ("raw", "post"):
            for metric_name in ("hd95_mm", "sensitivity", "specificity", "pred_voxels", "gt_voxels"):
                row[f"{mode}_{metric_name}_{class_name}"] = case_result[mode]["per_class"][class_name][metric_name]
            row[f"{mode}_empty_region_{class_name}"] = case_result[mode]["per_class"][class_name]["empty_region"]
    row["raw_hierarchy_et_outside_tc_voxels"] = case_result["raw"]["hierarchy"]["et_outside_tc_voxels"]
    row["raw_hierarchy_tc_outside_wt_voxels"] = case_result["raw"]["hierarchy"]["tc_outside_wt_voxels"]
    row["post_hierarchy_et_outside_tc_voxels"] = case_result["post"]["hierarchy"]["et_outside_tc_voxels"]
    row["post_hierarchy_tc_outside_wt_voxels"] = case_result["post"]["hierarchy"]["tc_outside_wt_voxels"]
    return row


def write_summary_csv(output_path, results):
    rows = [build_summary_row(item) for item in results]
    if not rows:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value):
    return f"{float(value):.4f}"


def _relative_path(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def write_summary_markdown(output_path, summary):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate = summary.get("aggregate", {})
    aggregate_wt_continuity = summary.get("aggregate_wt_continuity", {})
    aggregate_consistency = summary.get("aggregate_consistency", {})
    config = summary.get("config", {})
    cases = summary.get("cases", [])
    failures = summary.get("failures", [])
    output_root = Path(config["output_root"])

    lines = [
        "# Postprocess Validation Summary",
        "",
        "## Run Config",
        "",
        f"- Cases root: `{config['cases_root']}`",
        f"- Output root: `{config['output_root']}`",
        f"- Finetune method: `{config['finetune_method']}`",
        f"- Prompt mode: `{config['prompt_mode']}`",
        f"- Image size: `{config['image_size']}`",
        f"- Threshold: `{config['threshold']}`",
        f"- Device: `{config['device']}`",
        f"- Postprocess: `{json.dumps(config['postprocess'], ensure_ascii=False)}`",
        "",
    ]

    if config.get("prompt_mode") == "yolo_box":
        top2_rules = config.get("yolo_top2_rules") or {}
        box_strategy_by_class = config.get("prompt_box_strategy_by_class") or {}
        z_prompt = config.get("yolo_z_prompt") or {}
        wt_continuity = config.get("yolo_wt_continuity") or {}
        lines.extend([
            "### YOLO Prompt Config",
            "",
            f"- YOLO checkpoint: `{config['yolo_checkpoint']}`",
            f"- YOLO conf: `{config['yolo_conf']}`",
            f"- YOLO iou: `{config['yolo_iou']}`",
            f"- YOLO max_det: `{config['yolo_max_det']}`",
            f"- YOLO topk: `{config['yolo_topk']}`",
            f"- Prompt box strategy: `{config['prompt_box_strategy']}`",
            f"- Prompt box strategy ET: `{box_strategy_by_class.get('ET')}`",
            f"- Prompt box strategy TC: `{box_strategy_by_class.get('TC')}`",
            f"- Prompt box strategy WT: `{box_strategy_by_class.get('WT')}`",
            f"- Class prompt variant: `{config.get('class_prompt_variant')}`",
            f"- ET prompt variant: `{config.get('et_prompt_variant')}`",
            f"- Top2 score ratio: `{top2_rules.get('score_ratio')}`",
            f"- Top2 area ratio min: `{top2_rules.get('area_ratio_min')}`",
            f"- Top2 area ratio max: `{top2_rules.get('area_ratio_max')}`",
            f"- Top2 box IoU max: `{top2_rules.get('box_iou_max')}`",
            f"- Z prompt mode: `{z_prompt.get('mode')}`",
            f"- Z smooth window: `{z_prompt.get('smooth_window')}`",
            f"- Z fill gap max: `{z_prompt.get('fill_gap_max')}`",
            f"- Z center shift max: `{z_prompt.get('center_shift_max')}`",
            f"- Z area ratio min: `{z_prompt.get('area_ratio_min')}`",
            f"- Z area ratio max: `{z_prompt.get('area_ratio_max')}`",
            f"- WT continuity enabled: `{wt_continuity.get('enabled')}`",
            f"- WT continuity score thresh: `{wt_continuity.get('score_thresh')}`",
            f"- WT continuity center shift max: `{wt_continuity.get('center_shift_max')}`",
            f"- WT continuity area ratio min: `{wt_continuity.get('area_ratio_min')}`",
            f"- WT continuity area ratio max: `{wt_continuity.get('area_ratio_max')}`",
            f"- WT continuity mask dilate iters: `{wt_continuity.get('mask_dilate_iters')}`",
            f"- WT continuity mask blur kernel: `{wt_continuity.get('mask_blur_kernel')}`",
            "",
        ])

    if cases:
        lines.extend([
            "## Aggregate",
            "",
            "| Metric | Raw | Post | Delta |",
            "| --- | ---: | ---: | ---: |",
            f"| Mean Dice | {_format_metric(aggregate['raw']['mean_dice'])} | {_format_metric(aggregate['post']['mean_dice'])} | {_format_metric(aggregate['delta']['mean_dice'])} |",
            f"| Mean IoU | {_format_metric(aggregate['raw']['mean_iou'])} | {_format_metric(aggregate['post']['mean_iou'])} | {_format_metric(aggregate['delta']['mean_iou'])} |",
            "",
            "### Per-class Aggregate",
            "",
            "| Class | Raw Dice | Post Dice | Delta Dice | Raw IoU | Post IoU | Delta IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])

        for class_name in CLASS_NAMES:
            lines.append(
                f"| {class_name} | "
                f"{_format_metric(aggregate['raw']['per_class'][class_name]['dice'])} | "
                f"{_format_metric(aggregate['post']['per_class'][class_name]['dice'])} | "
                f"{_format_metric(aggregate['delta']['per_class'][class_name]['dice'])} | "
                f"{_format_metric(aggregate['raw']['per_class'][class_name]['iou'])} | "
                f"{_format_metric(aggregate['post']['per_class'][class_name]['iou'])} | "
                f"{_format_metric(aggregate['delta']['per_class'][class_name]['iou'])} |"
            )
        if aggregate_wt_continuity.get("num_cases_with_stats", 0) > 0:
            lines.extend([
                "",
                "### WT Continuity Aggregate",
                "",
                "| Eligible | Trigger | Rescue | Neutral | Harm | Trigger Reasons |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
                (
                    f"| {aggregate_wt_continuity.get('eligible_total', 0)} | "
                    f"{aggregate_wt_continuity.get('trigger_total', 0)} | "
                    f"{aggregate_wt_continuity.get('rescue', 0)} | "
                    f"{aggregate_wt_continuity.get('neutral', 0)} | "
                    f"{aggregate_wt_continuity.get('harm', 0)} | "
                    f"`{json.dumps(aggregate_wt_continuity.get('trigger_reasons', {}), ensure_ascii=False)}` |"
                ),
            ])
        raw_consistency = aggregate_consistency.get("raw") or {}
        post_consistency = aggregate_consistency.get("post") or {}
        prompt_sources = aggregate_consistency.get("prompt_sources") or {}
        lines.extend([
            "",
            "### Prompt Consistency Aggregate",
            "",
            "| Stage | WT=TC=ET cases | Ratio | ET<=TC<=WT valid cases | Ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| Raw | {raw_consistency.get('all_equal_cases', 0)} | "
                f"{_format_metric(raw_consistency.get('all_equal_ratio', 0.0))} | "
                f"{raw_consistency.get('hierarchy_order_valid_cases', 0)} | "
                f"{_format_metric(raw_consistency.get('hierarchy_order_valid_ratio', 0.0))} |"
            ),
            (
                f"| Post | {post_consistency.get('all_equal_cases', 0)} | "
                f"{_format_metric(post_consistency.get('all_equal_ratio', 0.0))} | "
                f"{post_consistency.get('hierarchy_order_valid_cases', 0)} | "
                f"{_format_metric(post_consistency.get('hierarchy_order_valid_ratio', 0.0))} |"
            ),
            "",
            "### Prompt Source Aggregate",
            "",
            "| Class | Source Counts |",
            "| --- | --- |",
        ])
        for class_name in CLASS_NAMES[::-1]:
            lines.append(f"| {class_name} | `{json.dumps(prompt_sources.get(class_name, {}), ensure_ascii=False)}` |")
    else:
        lines.extend([
            "## Aggregate",
            "",
            "No successful cases were processed.",
        ])

    lines.extend([
        "",
        "## Cases",
        "",
        "| Case | Raw Mean Dice | Post Mean Dice | Delta | Raw Mean IoU | Post Mean IoU | Delta | HTML |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])

    for case_result in cases:
        html_path = case_result["html_path"]
        preview_link = "-"
        if html_path:
            html_rel = _relative_path(html_path, output_root).replace(chr(92), "/")
            preview_link = f"[preview]({html_rel})"
        lines.append(
            f"| {case_result['case_id']} | "
            f"{_format_metric(case_result['raw']['mean_dice'])} | "
            f"{_format_metric(case_result['post']['mean_dice'])} | "
            f"{_format_metric(case_result['post']['mean_dice'] - case_result['raw']['mean_dice'])} | "
            f"{_format_metric(case_result['raw']['mean_iou'])} | "
            f"{_format_metric(case_result['post']['mean_iou'])} | "
            f"{_format_metric(case_result['post']['mean_iou'] - case_result['raw']['mean_iou'])} | "
            f"{preview_link} |"
        )

    for case_result in cases:
        lines.extend([
            "",
            f"### {case_result['case_id']}",
            "",
            "| Class | Raw Dice | Post Dice | Delta Dice | Raw IoU | Post IoU | Delta IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for class_name in CLASS_NAMES:
            raw_metrics = case_result["raw"]["per_class"][class_name]
            post_metrics = case_result["post"]["per_class"][class_name]
            lines.append(
                f"| {class_name} | "
                f"{_format_metric(raw_metrics['dice'])} | "
                f"{_format_metric(post_metrics['dice'])} | "
                f"{_format_metric(post_metrics['dice'] - raw_metrics['dice'])} | "
                f"{_format_metric(raw_metrics['iou'])} | "
                f"{_format_metric(post_metrics['iou'])} | "
                f"{_format_metric(post_metrics['iou'] - raw_metrics['iou'])} |"
            )
        if case_result["html_path"]:
            html_rel = _relative_path(case_result["html_path"], output_root).replace("\\", "/")
            lines.append("")
            lines.append(f"- 3D Preview: [{html_rel}]({html_rel})")
        wt_continuity = case_result.get("wt_continuity")
        if wt_continuity:
            wt_summary = wt_continuity.get("summary") or {}
            lines.extend([
                "",
                "| WT Continuity Eligible | Trigger | Rescue | Neutral | Harm | Trigger Reasons |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
                (
                    f"| {wt_summary.get('eligible_total', 0)} | "
                    f"{wt_summary.get('trigger_total', 0)} | "
                    f"{wt_summary.get('rescue', 0)} | "
                    f"{wt_summary.get('neutral', 0)} | "
                    f"{wt_summary.get('harm', 0)} | "
                    f"`{json.dumps(wt_summary.get('trigger_reasons', {}), ensure_ascii=False)}` |"
                ),
            ])

    if failures:
        lines.extend([
            "",
            "## Failures",
            "",
            "| Case | Error |",
            "| --- | --- |",
        ])
        for failure in failures:
            lines.append(f"| {failure['case_id']} | {failure['error']} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_case(case_dir, output_root, model, prompt_provider, device, args):
    brats_case = BraTSCase.from_dir(case_dir)
    output_dir = Path(output_root) / brats_case.case_id
    output_dir.mkdir(parents=True, exist_ok=True)

    class_volumes, inference_report = run_volume_inference(
        model=model,
        brats_case=brats_case,
        prompt_provider=prompt_provider,
        image_size=args.image_size,
        threshold=args.threshold,
        device=device,
        use_amp=args.use_amp,
        class_prompt_variant=args.class_prompt_variant,
        et_prompt_variant=args.et_prompt_variant,
    )
    raw_combined = build_combined_label(class_volumes)
    save_mask_outputs(brats_case, output_dir, class_volumes, raw_combined)

    prompt_report = None
    prompt_report_path = None

    postprocess_config = build_postprocess_config(
        enabled=args.postprocess,
        closing_radius=args.closing_radius,
        opening_radius=args.opening_radius,
        wt_keep_largest=args.wt_keep_largest,
        keep_topk_tc=args.keep_topk_tc,
        keep_topk_et=args.keep_topk_et,
        z_smooth_iterations=args.z_smooth_iterations,
    )

    postprocessed_volumes = class_volumes
    postprocess_report_path = None
    if args.postprocess:
        postprocessed_volumes, postprocess_report = postprocess_brats_masks(
            class_volumes=class_volumes,
            closing_radius=args.closing_radius,
            opening_radius=args.opening_radius,
            wt_keep_largest=args.wt_keep_largest,
            keep_topk_tc=args.keep_topk_tc,
            keep_topk_et=args.keep_topk_et,
            z_smooth_iterations=args.z_smooth_iterations,
        )
        post_combined = build_combined_label(postprocessed_volumes)
        save_mask_outputs(brats_case, output_dir, postprocessed_volumes, post_combined, prefix="post")

        postprocess_report = {
            "case_id": brats_case.case_id,
            "output_dir": str(output_dir.resolve()),
            **postprocess_report,
        }
        postprocess_report_path = output_dir / "postprocess_report.json"
        save_json(postprocess_report_path, postprocess_report)
    inference_report["post_consistency"] = analyze_class_volume_consistency(postprocessed_volumes)

    initial_meta = build_case_meta(
        brats_case=brats_case,
        output_dir=output_dir,
        prompt_mode=args.prompt_mode,
        finetune_method=args.finetune_method,
        sam_checkpoint=args.sam_checkpoint,
        finetuned_checkpoint=args.finetuned_checkpoint,
        image_size=args.image_size,
        threshold=args.threshold,
        postprocess_config=postprocess_config,
        postprocess_report_path=postprocess_report_path,
        prompt_report_path=prompt_report_path,
        yolo_config=build_yolo_prompt_config(
            prompt_mode=args.prompt_mode,
            yolo_checkpoint=args.yolo_checkpoint,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det,
            yolo_topk=args.yolo_topk,
            prompt_box_strategy=args.prompt_box_strategy,
            prompt_box_strategy_et=args.prompt_box_strategy_et,
            prompt_box_strategy_tc=args.prompt_box_strategy_tc,
            prompt_box_strategy_wt=args.prompt_box_strategy_wt,
            top2_score_ratio=args.top2_score_ratio,
            top2_area_ratio_min=args.top2_area_ratio_min,
            top2_area_ratio_max=args.top2_area_ratio_max,
            top2_iou_max=args.top2_iou_max,
            z_prompt_mode=args.z_prompt_mode,
            z_smooth_window=args.z_smooth_window,
            z_fill_gap_max=args.z_fill_gap_max,
            z_center_shift_max=args.z_center_shift_max,
            z_area_ratio_min=args.z_area_ratio_min,
            z_area_ratio_max=args.z_area_ratio_max,
            wt_continuity_enabled=args.wt_continuity_enabled,
            wt_continuity_score_thresh=args.wt_continuity_score_thresh,
            wt_continuity_center_shift_max=args.wt_continuity_center_shift_max,
            wt_continuity_area_ratio_min=args.wt_continuity_area_ratio_min,
            wt_continuity_area_ratio_max=args.wt_continuity_area_ratio_max,
            wt_continuity_mask_dilate_iters=args.wt_continuity_mask_dilate_iters,
            wt_continuity_mask_blur_kernel=args.wt_continuity_mask_blur_kernel,
            class_prompt_variant=args.class_prompt_variant,
            et_prompt_variant=args.et_prompt_variant,
        ),
    )
    save_case_meta(brats_case, output_dir, initial_meta)

    html_path = None
    if args.render_html:
        from visualize_case import render_case

        html_path, _ = render_case(
            output_dir=output_dir,
            mask_name=args.html_mask_name,
            save_path=None,
            show=False,
            opacity=args.html_opacity,
        )

    gt_masks, spacing_mm = load_ground_truth_masks(case_dir)
    raw_metrics = evaluate_brats_case(class_volumes, gt_masks, spacing_mm)
    post_metrics = evaluate_brats_case(postprocessed_volumes, gt_masks, spacing_mm)
    if hasattr(prompt_provider, "build_case_prompt_report"):
        prompt_report = prompt_provider.build_case_prompt_report(brats_case, gt_masks=gt_masks)
        prompt_report["config"] = build_yolo_prompt_config(
            prompt_mode=args.prompt_mode,
            yolo_checkpoint=args.yolo_checkpoint,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det,
            yolo_topk=args.yolo_topk,
            prompt_box_strategy=args.prompt_box_strategy,
            prompt_box_strategy_et=args.prompt_box_strategy_et,
            prompt_box_strategy_tc=args.prompt_box_strategy_tc,
            prompt_box_strategy_wt=args.prompt_box_strategy_wt,
            top2_score_ratio=args.top2_score_ratio,
            top2_area_ratio_min=args.top2_area_ratio_min,
            top2_area_ratio_max=args.top2_area_ratio_max,
            top2_iou_max=args.top2_iou_max,
            z_prompt_mode=args.z_prompt_mode,
            z_smooth_window=args.z_smooth_window,
            z_fill_gap_max=args.z_fill_gap_max,
            z_center_shift_max=args.z_center_shift_max,
            z_area_ratio_min=args.z_area_ratio_min,
            z_area_ratio_max=args.z_area_ratio_max,
            wt_continuity_enabled=args.wt_continuity_enabled,
            wt_continuity_score_thresh=args.wt_continuity_score_thresh,
            wt_continuity_center_shift_max=args.wt_continuity_center_shift_max,
            wt_continuity_area_ratio_min=args.wt_continuity_area_ratio_min,
            wt_continuity_area_ratio_max=args.wt_continuity_area_ratio_max,
            wt_continuity_mask_dilate_iters=args.wt_continuity_mask_dilate_iters,
            wt_continuity_mask_blur_kernel=args.wt_continuity_mask_blur_kernel,
            class_prompt_variant=args.class_prompt_variant,
            et_prompt_variant=args.et_prompt_variant,
        )
        prompt_report["runtime_prompt_summary"] = inference_report["prompt_summary"]
        prompt_report["runtime_prompt_events"] = inference_report["prompt_records"]
        prompt_report["mask_quality_checks"] = {
            "raw": inference_report["raw_consistency"],
            "post": inference_report.get("post_consistency"),
        }
        prompt_report_path = output_dir / "prompt_stats.json"
        save_json(prompt_report_path, prompt_report)

    meta = build_case_meta(
        brats_case=brats_case,
        output_dir=output_dir,
        prompt_mode=args.prompt_mode,
        finetune_method=args.finetune_method,
        sam_checkpoint=args.sam_checkpoint,
        finetuned_checkpoint=args.finetuned_checkpoint,
        image_size=args.image_size,
        threshold=args.threshold,
        postprocess_config=postprocess_config,
        postprocess_report_path=postprocess_report_path,
        prompt_report_path=prompt_report_path,
        yolo_config=build_yolo_prompt_config(
            prompt_mode=args.prompt_mode,
            yolo_checkpoint=args.yolo_checkpoint,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det,
            yolo_topk=args.yolo_topk,
            prompt_box_strategy=args.prompt_box_strategy,
            prompt_box_strategy_et=args.prompt_box_strategy_et,
            prompt_box_strategy_tc=args.prompt_box_strategy_tc,
            prompt_box_strategy_wt=args.prompt_box_strategy_wt,
            top2_score_ratio=args.top2_score_ratio,
            top2_area_ratio_min=args.top2_area_ratio_min,
            top2_area_ratio_max=args.top2_area_ratio_max,
            top2_iou_max=args.top2_iou_max,
            z_prompt_mode=args.z_prompt_mode,
            z_smooth_window=args.z_smooth_window,
            z_fill_gap_max=args.z_fill_gap_max,
            z_center_shift_max=args.z_center_shift_max,
            z_area_ratio_min=args.z_area_ratio_min,
            z_area_ratio_max=args.z_area_ratio_max,
            wt_continuity_enabled=args.wt_continuity_enabled,
            wt_continuity_score_thresh=args.wt_continuity_score_thresh,
            wt_continuity_center_shift_max=args.wt_continuity_center_shift_max,
            wt_continuity_area_ratio_min=args.wt_continuity_area_ratio_min,
            wt_continuity_area_ratio_max=args.wt_continuity_area_ratio_max,
            wt_continuity_mask_dilate_iters=args.wt_continuity_mask_dilate_iters,
            wt_continuity_mask_blur_kernel=args.wt_continuity_mask_blur_kernel,
            class_prompt_variant=args.class_prompt_variant,
            et_prompt_variant=args.et_prompt_variant,
        ),
    )
    save_case_meta(brats_case, output_dir, meta)
    return {
        "case_id": brats_case.case_id,
        "output_dir": str(output_dir.resolve()),
        "html_path": str(Path(html_path).resolve()) if html_path else None,
        "raw": raw_metrics,
        "post": post_metrics,
        "prompt_report_path": str(prompt_report_path.resolve()) if prompt_report_path else None,
        "wt_continuity": None if prompt_report is None else prompt_report.get("wt_continuity"),
        "raw_consistency": inference_report["raw_consistency"],
        "post_consistency": inference_report.get("post_consistency"),
        "prompt_summary": inference_report["prompt_summary"],
        "evaluation_grid": {
            "shape": list(brats_case.shape),
            "spacing_mm": list(spacing_mm),
            "prediction_resampled": False,
        },
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    case_dirs = select_case_dirs(args.cases_root, case_ids=args.case_ids, max_cases=args.max_cases)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)

    LOGGER.info("Loading model once for %d case(s).", len(case_dirs))
    model = load_multitask_model(
        model_type=args.model_type,
        image_size=args.image_size,
        sam_checkpoint=args.sam_checkpoint,
        finetune_method=args.finetune_method,
        finetuned_checkpoint=args.finetuned_checkpoint,
        device=device,
        input_channels=args.input_channels,
        encoder_adapter=args.encoder_adapter,
    )
    prompt_provider = build_prompt_provider(
        args.prompt_mode,
        args.image_size,
        yolo_checkpoint=args.yolo_checkpoint,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_max_det=args.yolo_max_det,
        yolo_topk=args.yolo_topk,
        prompt_box_strategy=args.prompt_box_strategy,
        prompt_box_strategy_et=args.prompt_box_strategy_et,
        prompt_box_strategy_tc=args.prompt_box_strategy_tc,
        prompt_box_strategy_wt=args.prompt_box_strategy_wt,
        top2_score_ratio=args.top2_score_ratio,
        top2_area_ratio_min=args.top2_area_ratio_min,
        top2_area_ratio_max=args.top2_area_ratio_max,
        top2_iou_max=args.top2_iou_max,
        z_prompt_mode=args.z_prompt_mode,
        z_smooth_window=args.z_smooth_window,
        z_fill_gap_max=args.z_fill_gap_max,
        z_center_shift_max=args.z_center_shift_max,
        z_area_ratio_min=args.z_area_ratio_min,
        z_area_ratio_max=args.z_area_ratio_max,
        wt_continuity_enabled=args.wt_continuity_enabled,
        wt_continuity_score_thresh=args.wt_continuity_score_thresh,
        wt_continuity_center_shift_max=args.wt_continuity_center_shift_max,
        wt_continuity_area_ratio_min=args.wt_continuity_area_ratio_min,
        wt_continuity_area_ratio_max=args.wt_continuity_area_ratio_max,
        wt_continuity_mask_dilate_iters=args.wt_continuity_mask_dilate_iters,
        wt_continuity_mask_blur_kernel=args.wt_continuity_mask_blur_kernel,
        device=args.device,
    )

    results = []
    failures = []
    for case_dir in case_dirs:
        LOGGER.info("Processing case %s", case_dir.name)
        try:
            result = run_single_case(
                case_dir=case_dir,
                output_root=output_root,
                model=model,
                prompt_provider=prompt_provider,
                device=device,
                args=args,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to process case %s", case_dir.name)
            failures.append({"case_id": case_dir.name, "error": str(exc)})

    summary = {
        "config": {
            "cases_root": str(Path(args.cases_root).resolve()),
            "output_root": str(output_root.resolve()),
            "case_ids": [path.name for path in case_dirs],
            "sam_checkpoint": str(Path(args.sam_checkpoint).resolve()),
            "finetuned_checkpoint": str(Path(args.finetuned_checkpoint).resolve()),
            "finetune_method": args.finetune_method,
            "prompt_mode": args.prompt_mode,
            "image_size": args.image_size,
            "threshold": args.threshold,
            "device": str(device),
            "use_amp": bool(args.use_amp),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "yolo_checkpoint": str(Path(args.yolo_checkpoint).resolve()) if args.prompt_mode == "yolo_box" else None,
            "yolo_conf": float(args.yolo_conf) if args.prompt_mode == "yolo_box" else None,
            "yolo_iou": float(args.yolo_iou) if args.prompt_mode == "yolo_box" else None,
            "yolo_max_det": int(args.yolo_max_det) if args.prompt_mode == "yolo_box" else None,
            "yolo_topk": int(args.yolo_topk) if args.prompt_mode == "yolo_box" else None,
            "prompt_box_strategy": str(args.prompt_box_strategy) if args.prompt_mode == "yolo_box" else None,
            "prompt_box_strategy_by_class": {
                "ET": str(args.prompt_box_strategy_et or args.prompt_box_strategy),
                "TC": str(args.prompt_box_strategy_tc or args.prompt_box_strategy),
                "WT": str(args.prompt_box_strategy_wt or args.prompt_box_strategy),
            } if args.prompt_mode == "yolo_box" else None,
            "class_prompt_variant": str(args.class_prompt_variant) if args.prompt_mode == "yolo_box" else None,
            "et_prompt_variant": str(args.et_prompt_variant) if args.prompt_mode == "yolo_box" else None,
            "yolo_top2_rules": {
                "score_ratio": float(args.top2_score_ratio),
                "area_ratio_min": float(args.top2_area_ratio_min),
                "area_ratio_max": float(args.top2_area_ratio_max),
                "box_iou_max": float(args.top2_iou_max),
            } if args.prompt_mode == "yolo_box" else None,
            "yolo_z_prompt": {
                "mode": str(args.z_prompt_mode),
                "smooth_window": int(args.z_smooth_window),
                "fill_gap_max": int(args.z_fill_gap_max),
                "center_shift_max": float(args.z_center_shift_max),
                "area_ratio_min": float(args.z_area_ratio_min),
                "area_ratio_max": float(args.z_area_ratio_max),
            } if args.prompt_mode == "yolo_box" else None,
            "yolo_wt_continuity": {
                "enabled": bool(args.wt_continuity_enabled),
                "score_thresh": float(args.wt_continuity_score_thresh),
                "center_shift_max": float(args.wt_continuity_center_shift_max),
                "area_ratio_min": float(args.wt_continuity_area_ratio_min),
                "area_ratio_max": float(args.wt_continuity_area_ratio_max),
                "mask_dilate_iters": int(args.wt_continuity_mask_dilate_iters),
                "mask_blur_kernel": int(args.wt_continuity_mask_blur_kernel),
            } if args.prompt_mode == "yolo_box" else None,
            "postprocess": build_postprocess_config(
                enabled=args.postprocess,
                closing_radius=args.closing_radius,
                opening_radius=args.opening_radius,
                wt_keep_largest=args.wt_keep_largest,
                keep_topk_tc=args.keep_topk_tc,
                keep_topk_et=args.keep_topk_et,
                z_smooth_iterations=args.z_smooth_iterations,
            ),
            "html_mask_name": args.html_mask_name,
            "render_html": bool(args.render_html),
            "metric_contract": {
                "schema_version": 1,
                "grid_policy": "Predictions must match the native ground-truth grid; this SAM path writes native-grid masks.",
                "empty_mask_policy": "Both-empty Dice/IoU are 1; one-sided empty Dice/IoU are 0; HD95 is null when a surface is absent.",
            },
        },
        "aggregate": summarize_results(results),
        "aggregate_wt_continuity": summarize_wt_continuity(results),
        "aggregate_consistency": {
            "raw": summarize_consistency_across_cases(results, "raw_consistency"),
            "post": summarize_consistency_across_cases(results, "post_consistency"),
            "prompt_sources": merge_prompt_source_counts(results),
        },
        "cases": results,
        "failures": failures,
    }

    save_json(output_root / "summary_metrics.json", summary)
    write_summary_csv(output_root / "summary_metrics.csv", results)
    write_summary_markdown(output_root / "summary.md", summary)

    print(json.dumps({
        "processed_cases": len(results),
        "failed_cases": len(failures),
        "output_root": str(output_root.resolve()),
        "summary_json": str((output_root / "summary_metrics.json").resolve()),
        "summary_csv": str((output_root / "summary_metrics.csv").resolve()),
        "summary_md": str((output_root / "summary.md").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
