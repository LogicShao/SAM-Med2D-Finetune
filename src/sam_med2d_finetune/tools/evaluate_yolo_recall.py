import argparse
import csv
import hashlib
import json
from pathlib import Path

from sam_med2d_finetune.tools.train_yolo import (
    configure_ultralytics_env,
    normalize_data_yaml,
    resolve_data_yaml,
)


DEFAULT_CONF_VALUES = "0.001,0.003,0.005,0.01,0.03,0.05,0.10"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO checkpoint for BraTS slice-level recall under multiple confidence thresholds."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a YOLO checkpoint, usually best.pt or last.pt.",
    )
    parser.add_argument(
        "--data",
        default="datasets/brats_yolo_dev",
        help="Path to data.yaml or a dataset directory containing data.yaml.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--conf_values",
        default=DEFAULT_CONF_VALUES,
        help="Comma-separated confidence thresholds to scan.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.60,
        help="Default NMS IoU threshold used during prediction when --iou_values is not set.",
    )
    parser.add_argument(
        "--iou_values",
        default="",
        help="Optional comma-separated IoU thresholds to scan. Overrides --iou when provided.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image size for YOLO predict.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device, e.g. 0 or cpu.",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=1,
        help="Maximum number of detections per image.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Predict batch size.",
    )
    parser.add_argument(
        "--ultralytics_dir",
        default=".ultralytics",
        help="Writable directory for Ultralytics settings/cache files.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/yolo_recall_scan",
        help="Directory to save the scan summary.",
    )
    return parser.parse_args()


def parse_float_values(raw_value, option_name):
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError(f"No valid {option_name} values were provided.")
    return values


def parse_conf_values(raw_value):
    return parse_float_values(raw_value, "--conf_values")


def parse_iou_values(raw_value, fallback_iou):
    if not raw_value.strip():
        return [float(fallback_iou)]
    return parse_float_values(raw_value, "--iou_values")


def load_dataset_paths(data_arg, ultralytics_dir):
    data_yaml = resolve_data_yaml(data_arg)
    normalized_yaml = normalize_data_yaml(data_yaml, ultralytics_dir)
    config = json.loads(json.dumps(yaml_safe_load(normalized_yaml)))
    dataset_root = Path(config["path"]).resolve()
    dataset_manifest = dataset_root / "dataset_manifest.json"
    return {
        "data_yaml": data_yaml,
        "normalized_yaml": normalized_yaml,
        "dataset_root": dataset_root,
        "dataset_manifest": dataset_manifest if dataset_manifest.is_file() else None,
        "dataset_manifest_sha256": sha256_file(dataset_manifest) if dataset_manifest.is_file() else None,
    }


def yaml_safe_load(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_split_dirs(dataset_root, split_name):
    image_dir = Path(dataset_root) / "images" / split_name
    label_dir = Path(dataset_root) / "labels" / split_name
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image split directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label split directory not found: {label_dir}")
    return image_dir, label_dir


def load_ground_truth(label_dir):
    gt_by_stem = {}
    for label_path in sorted(Path(label_dir).glob("*.txt")):
        lines = [
            line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        boxes = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label line in {label_path}: {line}")
            _, xc, yc, w, h = parts
            boxes.append((float(xc), float(yc), float(w), float(h)))
        if boxes:
            gt_by_stem[label_path.stem] = boxes
    return gt_by_stem


def yolo_xyxy_to_normalized_xywh(box_xyxy, image_width, image_height):
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return (
        x_center / max(float(image_width), 1.0),
        y_center / max(float(image_height), 1.0),
        width / max(float(image_width), 1.0),
        height / max(float(image_height), 1.0),
    )


def normalized_xywh_to_xyxy(box_xywh):
    x_center, y_center, width, height = box_xywh
    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0
    return x1, y1, x2, y2


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = normalized_xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = normalized_xywh_to_xyxy(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def bbox_intersection_over_gt(predicted_box, gt_box):
    px1, py1, px2, py2 = normalized_xywh_to_xyxy(predicted_box)
    gx1, gy1, gx2, gy2 = normalized_xywh_to_xyxy(gt_box)
    inter_width = max(0.0, min(px2, gx2) - max(px1, gx1))
    inter_height = max(0.0, min(py2, gy2) - max(py1, gy1))
    gt_area = max(gx2 - gx1, 0.0) * max(gy2 - gy1, 0.0)
    if gt_area <= 0.0:
        return 0.0
    return inter_width * inter_height / gt_area


def bbox_area_ratio(predicted_box, gt_box):
    px1, py1, px2, py2 = normalized_xywh_to_xyxy(predicted_box)
    gx1, gy1, gx2, gy2 = normalized_xywh_to_xyxy(gt_box)
    predicted_area = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
    gt_area = max(gx2 - gx1, 0.0) * max(gy2 - gy1, 0.0)
    return predicted_area / gt_area if gt_area > 0.0 else 0.0


def prediction_xywh(prediction):
    if isinstance(prediction, dict):
        return tuple(float(value) for value in prediction["xywh"])
    return tuple(float(value) for value in prediction)


def parse_slice_stem(stem):
    case_id, separator, z_value = stem.rpartition("_z")
    if not separator or not case_id or not z_value.isdigit():
        raise ValueError(f"Invalid BraTS YOLO slice stem: {stem}")
    return case_id, int(z_value)


def longest_consecutive_misses(slice_results, key):
    longest = 0
    current = 0
    previous_z = None
    for item in sorted(slice_results, key=lambda entry: entry["z_index"]):
        z_index = item["z_index"]
        if previous_z is not None and z_index != previous_z + 1:
            current = 0
        if item[key]:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
        previous_z = z_index
    return longest


def best_positive_slice_match(gt_boxes, predictions):
    best_iou = 0.0
    best_coverage = 0.0
    best_area_ratio = None
    for gt_box in gt_boxes:
        for prediction in predictions:
            predicted_box = prediction_xywh(prediction)
            current_iou = bbox_iou(gt_box, predicted_box)
            current_coverage = bbox_intersection_over_gt(predicted_box, gt_box)
            best_iou = max(best_iou, current_iou)
            if best_area_ratio is None or current_coverage > best_coverage:
                best_coverage = current_coverage
                best_area_ratio = bbox_area_ratio(predicted_box, gt_box)
    return best_iou, best_coverage, best_area_ratio


def evaluate_predictions(predictions_by_stem, gt_by_stem, image_stems):
    positive_stems = [stem for stem in image_stems if stem in gt_by_stem]
    negative_stems = [stem for stem in image_stems if stem not in gt_by_stem]

    positive_count = len(positive_stems)
    negative_count = len(negative_stems)
    hit_any = 0
    hit_iou_01 = 0
    hit_iou_03 = 0
    hit_iou_05 = 0
    hit_coverage_05 = 0
    hit_coverage_08 = 0
    false_positive_slices = 0
    total_positive_boxes = 0
    total_negative_boxes = 0
    matched_area_ratios = []
    positive_results_by_case = {}

    for stem in positive_stems:
        predicted_boxes = predictions_by_stem.get(stem, [])
        total_positive_boxes += len(predicted_boxes)
        if predicted_boxes:
            hit_any += 1
        gt_boxes = gt_by_stem[stem]
        best_iou, best_coverage, best_area_ratio = best_positive_slice_match(gt_boxes, predicted_boxes)
        if best_iou >= 0.10:
            hit_iou_01 += 1
        if best_iou >= 0.30:
            hit_iou_03 += 1
        if best_iou >= 0.50:
            hit_iou_05 += 1
        if best_coverage >= 0.50:
            hit_coverage_05 += 1
        if best_coverage >= 0.80:
            hit_coverage_08 += 1
        if best_area_ratio is not None:
            matched_area_ratios.append(best_area_ratio)

        case_id, z_index = parse_slice_stem(stem)
        positive_results_by_case.setdefault(case_id, []).append({
            "z_index": z_index,
            "has_box": bool(predicted_boxes),
            "best_iou": best_iou,
            "best_coverage": best_coverage,
            "missed_any_box": not bool(predicted_boxes),
            "missed_coverage_0.50": best_coverage < 0.50,
        })

    for stem in negative_stems:
        predicted_boxes = predictions_by_stem.get(stem, [])
        total_negative_boxes += len(predicted_boxes)
        if predicted_boxes:
            false_positive_slices += 1

    def safe_div(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    per_case = []
    for case_id, slice_results in sorted(positive_results_by_case.items()):
        coverage_hit_count = sum(item["best_coverage"] >= 0.50 for item in slice_results)
        any_box_count = sum(item["has_box"] for item in slice_results)
        per_case.append({
            "case_id": case_id,
            "positive_slice_count": len(slice_results),
            "any_box_hit_count": any_box_count,
            "coverage_0.50_hit_count": coverage_hit_count,
            "coverage_0.50_recall": safe_div(coverage_hit_count, len(slice_results)),
            "fully_missed_any_box": any_box_count == 0,
            "fully_missed_coverage_0.50": coverage_hit_count == 0,
            "max_consecutive_any_box_misses": longest_consecutive_misses(
                slice_results,
                "missed_any_box",
            ),
            "max_consecutive_coverage_0.50_misses": longest_consecutive_misses(
                slice_results,
                "missed_coverage_0.50",
            ),
        })

    fully_missed_case_ids = [
        item["case_id"] for item in per_case if item["fully_missed_coverage_0.50"]
    ]
    max_consecutive_misses = max(
        (item["max_consecutive_coverage_0.50_misses"] for item in per_case),
        default=0,
    )

    return {
        "num_positive_slices": positive_count,
        "num_negative_slices": negative_count,
        "slice_recall_any_box": safe_div(hit_any, positive_count),
        "slice_recall_iou_0.10": safe_div(hit_iou_01, positive_count),
        "slice_recall_iou_0.30": safe_div(hit_iou_03, positive_count),
        "slice_recall_iou_0.50": safe_div(hit_iou_05, positive_count),
        "slice_coverage_recall_0.50": safe_div(hit_coverage_05, positive_count),
        "slice_coverage_recall_0.80": safe_div(hit_coverage_08, positive_count),
        "missed_positive_slice_count_coverage_0.50": positive_count - hit_coverage_05,
        "fully_missed_case_count": len(fully_missed_case_ids),
        "fully_missed_case_ids": fully_missed_case_ids,
        "max_consecutive_missed_positive_slices": max_consecutive_misses,
        "background_false_positive_rate": safe_div(false_positive_slices, negative_count),
        "avg_boxes_per_positive_slice": safe_div(total_positive_boxes, positive_count),
        "avg_boxes_per_negative_slice": safe_div(total_negative_boxes, negative_count),
        "mean_predicted_gt_box_area_ratio": safe_div(
            sum(matched_area_ratios),
            len(matched_area_ratios),
        ),
        "hit_any_count": hit_any,
        "hit_iou_0.10_count": hit_iou_01,
        "hit_iou_0.30_count": hit_iou_03,
        "hit_iou_0.50_count": hit_iou_05,
        "false_positive_slices": false_positive_slices,
        "per_case": per_case,
    }


def run_predict(model, image_dir, conf, iou, imgsz, device, max_det, batch):
    predictions = {}
    results = model.predict(
        source=str(image_dir),
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device,
        max_det=int(max_det),
        batch=int(batch),
        save=False,
        stream=True,
        verbose=False,
    )
    for result in results:
        stem = Path(result.path).stem
        boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            image_height, image_width = result.orig_shape
            xyxy_values = result.boxes.xyxy.cpu().tolist()
            confidence_values = result.boxes.conf.cpu().tolist()
            class_values = result.boxes.cls.cpu().tolist()
            for box_xyxy, confidence, class_id in zip(
                xyxy_values,
                confidence_values,
                class_values,
            ):
                boxes.append({
                    "xywh": list(
                        yolo_xyxy_to_normalized_xywh(box_xyxy, image_width, image_height)
                    ),
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                })
        predictions[stem] = boxes
    return predictions


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_token(value):
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def write_prediction_export(out_dir, predictions, image_stems, metadata):
    prediction_dir = Path(out_dir) / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"iou_{_float_token(metadata['iou'])}_conf_{_float_token(metadata['conf'])}.json"
    prediction_path = prediction_dir / file_name
    slices = []
    case_ids = set()
    for stem in image_stems:
        case_id, z_index = parse_slice_stem(stem)
        case_ids.add(case_id)
        slices.append({
            "stem": stem,
            "case_id": case_id,
            "z_index": z_index,
            "boxes": predictions.get(stem, []),
        })
    payload = {
        "schema_version": 1,
        **metadata,
        "case_count": len(case_ids),
        "case_ids": sorted(case_ids),
        "slice_count": len(slices),
        "slices": slices,
    }
    prediction_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return prediction_path, sha256_file(prediction_path)


def write_scan_outputs(out_dir, payload):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "scan_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / "scan_summary.csv"
    rows = payload["results"]
    fieldnames = [
        "iou",
        "conf",
        "slice_recall_any_box",
        "slice_recall_iou_0.10",
        "slice_recall_iou_0.30",
        "slice_recall_iou_0.50",
        "slice_coverage_recall_0.50",
        "slice_coverage_recall_0.80",
        "missed_positive_slice_count_coverage_0.50",
        "fully_missed_case_count",
        "max_consecutive_missed_positive_slices",
        "background_false_positive_rate",
        "avg_boxes_per_positive_slice",
        "avg_boxes_per_negative_slice",
        "mean_predicted_gt_box_area_ratio",
        "hit_any_count",
        "hit_iou_0.10_count",
        "hit_iou_0.30_count",
        "hit_iou_0.50_count",
        "false_positive_slices",
        "num_positive_slices",
        "num_negative_slices",
        "prediction_file",
        "prediction_sha256",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    markdown_path = out_dir / "scan_summary.md"
    markdown_path.write_text(build_scan_markdown(payload), encoding="utf-8")

    return json_path, csv_path, markdown_path


def choose_recommendation(results):
    if not results:
        return None

    sorted_results = sorted(
        results,
        key=lambda item: (
            item["fully_missed_case_count"],
            item["missed_positive_slice_count_coverage_0.50"],
            item["max_consecutive_missed_positive_slices"],
            -item["slice_coverage_recall_0.50"],
            item["background_false_positive_rate"],
            -item["mean_predicted_gt_box_area_ratio"],
            item["iou"],
            item["conf"],
        ),
    )
    return sorted_results[0]


def choose_topk(results, top_k=2):
    if top_k <= 0:
        return []
    sorted_results = sorted(
        results,
        key=lambda item: (
            item["fully_missed_case_count"],
            item["missed_positive_slice_count_coverage_0.50"],
            item["max_consecutive_missed_positive_slices"],
            -item["slice_coverage_recall_0.50"],
            item["background_false_positive_rate"],
            -item["mean_predicted_gt_box_area_ratio"],
            item["iou"],
            item["conf"],
        ),
    )
    return sorted_results[:top_k]


def _format_metric(value):
    return f"{float(value):.4f}"


def build_scan_markdown(payload):
    recommendation = payload.get("recommended")
    shortlist = payload.get("recommended_topk", [])
    lines = [
        "# YOLO Recall Scan Summary",
        "",
        "## Run Config",
        "",
        f"- Model: `{payload['model']}`",
        f"- Dataset root: `{payload['dataset_root']}`",
        f"- Split: `{payload['split']}`",
        f"- Image size: `{payload['imgsz']}`",
        f"- IoU values: `{', '.join(_format_metric(value) for value in payload['iou_values'])}`",
        f"- Confidence values: `{', '.join(_format_metric(value) for value in payload['conf_values'])}`",
        f"- Max detections: `{payload['max_det']}`",
        f"- Batch: `{payload['batch']}`",
        f"- Device: `{payload['device']}`",
        "",
    ]

    if recommendation is not None:
        lines.extend([
            "## Recommended",
            "",
            f"- IoU: `{_format_metric(recommendation['iou'])}`",
            f"- Conf: `{_format_metric(recommendation['conf'])}`",
            f"- Fully missed cases: `{recommendation['fully_missed_case_count']}`",
            "- Missed positive slices at coverage 0.50: "
            f"`{recommendation['missed_positive_slice_count_coverage_0.50']}`",
            f"- Maximum consecutive misses: `{recommendation['max_consecutive_missed_positive_slices']}`",
            f"- Slice coverage recall @0.50: `{_format_metric(recommendation['slice_coverage_recall_0.50'])}`",
            f"- Background false positive rate: `{_format_metric(recommendation['background_false_positive_rate'])}`",
            "",
        ])

    if shortlist:
        lines.extend([
            "## Top Candidates",
            "",
            "| Rank | IoU | Conf | Fully Missed Cases | Missed Slices | "
            "Max Consecutive | Coverage@0.50 | BG FP Rate | Area Ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for index, item in enumerate(shortlist, start=1):
            lines.append(
                f"| {index} | "
                f"{_format_metric(item['iou'])} | "
                f"{_format_metric(item['conf'])} | "
                f"{item['fully_missed_case_count']} | "
                f"{item['missed_positive_slice_count_coverage_0.50']} | "
                f"{item['max_consecutive_missed_positive_slices']} | "
                f"{_format_metric(item['slice_coverage_recall_0.50'])} | "
                f"{_format_metric(item['background_false_positive_rate'])} | "
                f"{_format_metric(item['mean_predicted_gt_box_area_ratio'])} |"
            )
        lines.append("")

    lines.extend([
        "## Full Grid",
        "",
        "| IoU | Conf | Fully Missed Cases | Missed Slices | Max Consecutive | "
        "Coverage@0.50 | Coverage@0.80 | BG FP Rate | Area Ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in payload["results"]:
        lines.append(
            f"| {_format_metric(item['iou'])} | "
            f"{_format_metric(item['conf'])} | "
            f"{item['fully_missed_case_count']} | "
            f"{item['missed_positive_slice_count_coverage_0.50']} | "
            f"{item['max_consecutive_missed_positive_slices']} | "
            f"{_format_metric(item['slice_coverage_recall_0.50'])} | "
            f"{_format_metric(item['slice_coverage_recall_0.80'])} | "
            f"{_format_metric(item['background_false_positive_rate'])} | "
            f"{_format_metric(item['mean_predicted_gt_box_area_ratio'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    conf_values = parse_conf_values(args.conf_values)
    iou_values = parse_iou_values(args.iou_values, args.iou)
    ultralytics_dir = configure_ultralytics_env(args.ultralytics_dir)
    dataset_info = load_dataset_paths(args.data, ultralytics_dir)
    image_dir, label_dir = resolve_split_dirs(dataset_info["dataset_root"], args.split)

    image_paths = sorted(image_dir.glob("*.png"))
    image_stems = [path.stem for path in image_paths]
    gt_by_stem = load_ground_truth(label_dir)
    if len(image_stems) != len(set(image_stems)):
        raise ValueError(f"Duplicate image stems found in split: {args.split}")

    from ultralytics import YOLO

    model = YOLO(str(Path(args.model).resolve()))
    model_sha256 = sha256_file(args.model)

    model_name = Path(args.model).stem
    out_dir = Path(args.out_dir) / f"{model_name}_{args.split}"
    results = []
    for iou in iou_values:
        for conf in conf_values:
            predictions = run_predict(
                model=model,
                image_dir=image_dir,
                conf=conf,
                iou=iou,
                imgsz=args.imgsz,
                device=args.device,
                max_det=args.max_det,
                batch=args.batch,
            )
            metrics = evaluate_predictions(predictions, gt_by_stem, image_stems)
            metrics["iou"] = float(iou)
            metrics["conf"] = float(conf)
            prediction_path, prediction_sha256 = write_prediction_export(
                out_dir,
                predictions,
                image_stems,
                {
                    "model": str(Path(args.model).resolve()),
                    "model_sha256": model_sha256,
                    "dataset_root": str(dataset_info["dataset_root"]),
                    "dataset_manifest": str(dataset_info["dataset_manifest"])
                    if dataset_info["dataset_manifest"]
                    else None,
                    "dataset_manifest_sha256": dataset_info["dataset_manifest_sha256"],
                    "split": args.split,
                    "imgsz": int(args.imgsz),
                    "max_det": int(args.max_det),
                    "iou": float(iou),
                    "conf": float(conf),
                },
            )
            metrics["prediction_file"] = str(prediction_path.resolve())
            metrics["prediction_sha256"] = prediction_sha256
            results.append(metrics)

    recommendation = choose_recommendation(results)
    recommended_topk = choose_topk(results)

    payload = {
        "model": str(Path(args.model).resolve()),
        "model_sha256": model_sha256,
        "data_yaml": str(dataset_info["data_yaml"]),
        "normalized_data_yaml": str(dataset_info["normalized_yaml"]),
        "dataset_root": str(dataset_info["dataset_root"]),
        "dataset_manifest": str(dataset_info["dataset_manifest"])
        if dataset_info["dataset_manifest"]
        else None,
        "dataset_manifest_sha256": dataset_info["dataset_manifest_sha256"],
        "split": args.split,
        "imgsz": int(args.imgsz),
        "iou_values": [float(value) for value in iou_values],
        "conf_values": [float(value) for value in conf_values],
        "max_det": int(args.max_det),
        "batch": int(args.batch),
        "device": args.device,
        "results": results,
        "recommended": recommendation,
        "recommended_topk": recommended_topk,
    }
    json_path, csv_path, markdown_path = write_scan_outputs(out_dir, payload)

    print(json.dumps({
        "out_dir": str(out_dir.resolve()),
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(markdown_path.resolve()),
        "recommended": recommendation,
        "recommended_topk": recommended_topk,
    }, indent=2))


if __name__ == "__main__":
    main()
