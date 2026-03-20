import argparse
import csv
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from brats_case import BraTSCase
from infer_volume import (
    CLASS_NAMES,
    DEFAULT_YOLO_CHECKPOINT,
    DEFAULT_YOLO_IMGSZ,
    build_case_meta,
    build_combined_label,
    build_postprocess_config,
    build_prompt_provider,
    run_volume_inference,
    save_case_meta,
    save_json,
    save_mask_outputs,
    str_to_bool,
)
from model_factory import load_multitask_model
from postprocess_3d import postprocess_brats_masks
from visualize_case import render_case


LOGGER = logging.getLogger(__name__)
EPSILON = 1e-7


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
    parser.add_argument("--yolo_checkpoint", default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--yolo_conf", type=float, default=0.05)
    parser.add_argument("--yolo_iou", type=float, default=0.60)
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

    seg_volume = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.int16)
    return {
        "ET": (seg_volume == 4).astype(np.uint8),
        "TC": np.isin(seg_volume, [1, 4]).astype(np.uint8),
        "WT": np.isin(seg_volume, [1, 2, 4]).astype(np.uint8),
    }


def compute_binary_metrics(pred_mask, gt_mask):
    pred = np.asarray(pred_mask).astype(bool, copy=False)
    gt = np.asarray(gt_mask).astype(bool, copy=False)

    intersection = int(np.count_nonzero(pred & gt))
    pred_voxels = int(np.count_nonzero(pred))
    gt_voxels = int(np.count_nonzero(gt))
    union = pred_voxels + gt_voxels - intersection

    dice = (2.0 * intersection + EPSILON) / (pred_voxels + gt_voxels + EPSILON)
    iou = (intersection + EPSILON) / (union + EPSILON)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "intersection": intersection,
    }


def evaluate_case_metrics(class_volumes, gt_masks):
    per_class = {}
    for class_name in CLASS_NAMES:
        per_class[class_name] = compute_binary_metrics(class_volumes[class_name], gt_masks[class_name])

    mean_dice = float(np.mean([per_class[class_name]["dice"] for class_name in CLASS_NAMES]))
    mean_iou = float(np.mean([per_class[class_name]["iou"] for class_name in CLASS_NAMES]))
    return {
        "per_class": per_class,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
    }


def summarize_results(results):
    if not results:
        return {"num_cases": 0}

    summary = {
        "num_cases": len(results),
        "raw": {"mean_dice": 0.0, "mean_iou": 0.0, "per_class": {}},
        "post": {"mean_dice": 0.0, "mean_iou": 0.0, "per_class": {}},
        "delta": {"mean_dice": 0.0, "mean_iou": 0.0, "per_class": {}},
    }

    for mode in ("raw", "post"):
        summary[mode]["mean_dice"] = float(np.mean([item[mode]["mean_dice"] for item in results]))
        summary[mode]["mean_iou"] = float(np.mean([item[mode]["mean_iou"] for item in results]))
        for class_name in CLASS_NAMES:
            summary[mode]["per_class"][class_name] = {
                "dice": float(np.mean([item[mode]["per_class"][class_name]["dice"] for item in results])),
                "iou": float(np.mean([item[mode]["per_class"][class_name]["iou"] for item in results])),
            }

    summary["delta"]["mean_dice"] = summary["post"]["mean_dice"] - summary["raw"]["mean_dice"]
    summary["delta"]["mean_iou"] = summary["post"]["mean_iou"] - summary["raw"]["mean_iou"]
    for class_name in CLASS_NAMES:
        summary["delta"]["per_class"][class_name] = {
            "dice": summary["post"]["per_class"][class_name]["dice"] - summary["raw"]["per_class"][class_name]["dice"],
            "iou": summary["post"]["per_class"][class_name]["iou"] - summary["raw"]["per_class"][class_name]["iou"],
        }
    return summary


def build_summary_row(case_result):
    row = {
        "case_id": case_result["case_id"],
        "output_dir": case_result["output_dir"],
        "html_path": case_result["html_path"],
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
        lines.extend([
            "### YOLO Prompt Config",
            "",
            f"- YOLO checkpoint: `{config['yolo_checkpoint']}`",
            f"- YOLO conf: `{config['yolo_conf']}`",
            f"- YOLO iou: `{config['yolo_iou']}`",
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
        html_rel = _relative_path(case_result["html_path"], output_root)
        lines.append(
            f"| {case_result['case_id']} | "
            f"{_format_metric(case_result['raw']['mean_dice'])} | "
            f"{_format_metric(case_result['post']['mean_dice'])} | "
            f"{_format_metric(case_result['post']['mean_dice'] - case_result['raw']['mean_dice'])} | "
            f"{_format_metric(case_result['raw']['mean_iou'])} | "
            f"{_format_metric(case_result['post']['mean_iou'])} | "
            f"{_format_metric(case_result['post']['mean_iou'] - case_result['raw']['mean_iou'])} | "
            f"[preview]({html_rel.replace(chr(92), '/')}) |"
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
        html_rel = _relative_path(case_result["html_path"], output_root).replace("\\", "/")
        lines.append("")
        lines.append(f"- 3D Preview: [{html_rel}]({html_rel})")

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

    class_volumes = run_volume_inference(
        model=model,
        brats_case=brats_case,
        prompt_provider=prompt_provider,
        image_size=args.image_size,
        threshold=args.threshold,
        device=device,
        use_amp=args.use_amp,
    )
    raw_combined = build_combined_label(class_volumes)
    save_mask_outputs(brats_case, output_dir, class_volumes, raw_combined)

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
        yolo_config={
            "checkpoint": str(Path(args.yolo_checkpoint).resolve()) if args.prompt_mode == "yolo_box" else None,
            "conf": float(args.yolo_conf) if args.prompt_mode == "yolo_box" else None,
            "iou": float(args.yolo_iou) if args.prompt_mode == "yolo_box" else None,
            "imgsz": DEFAULT_YOLO_IMGSZ if args.prompt_mode == "yolo_box" else None,
        },
    )
    save_case_meta(brats_case, output_dir, meta)

    html_path, _ = render_case(
        output_dir=output_dir,
        mask_name=args.html_mask_name,
        save_path=None,
        show=False,
        opacity=args.html_opacity,
    )

    gt_masks = load_ground_truth_masks(case_dir)
    raw_metrics = evaluate_case_metrics(class_volumes, gt_masks)
    post_metrics = evaluate_case_metrics(postprocessed_volumes, gt_masks)
    return {
        "case_id": brats_case.case_id,
        "output_dir": str(output_dir.resolve()),
        "html_path": str(Path(html_path).resolve()),
        "raw": raw_metrics,
        "post": post_metrics,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    case_dirs = select_case_dirs(args.cases_root, case_ids=args.case_ids, max_cases=args.max_cases)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

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
            "yolo_checkpoint": str(Path(args.yolo_checkpoint).resolve()) if args.prompt_mode == "yolo_box" else None,
            "yolo_conf": float(args.yolo_conf) if args.prompt_mode == "yolo_box" else None,
            "yolo_iou": float(args.yolo_iou) if args.prompt_mode == "yolo_box" else None,
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
        },
        "aggregate": summarize_results(results),
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
