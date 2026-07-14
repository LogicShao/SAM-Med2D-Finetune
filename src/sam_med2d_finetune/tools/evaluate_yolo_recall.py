import argparse
import csv
import json
from pathlib import Path

from sam_med2d_finetune.tools.train_yolo import (
    configure_ultralytics_env,
    normalize_data_yaml,
    resolve_data_yaml,
)


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
        default="0.05,0.10,0.15,0.20,0.25,0.30",
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
        default=20,
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
    return {
        "data_yaml": data_yaml,
        "normalized_yaml": normalized_yaml,
        "dataset_root": dataset_root,
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


def evaluate_predictions(predictions_by_stem, gt_by_stem, image_stems):
    positive_stems = [stem for stem in image_stems if stem in gt_by_stem]
    negative_stems = [stem for stem in image_stems if stem not in gt_by_stem]

    positive_count = len(positive_stems)
    negative_count = len(negative_stems)
    hit_any = 0
    hit_iou_01 = 0
    hit_iou_03 = 0
    hit_iou_05 = 0
    false_positive_slices = 0
    total_positive_boxes = 0
    total_negative_boxes = 0

    for stem in positive_stems:
        predicted_boxes = predictions_by_stem.get(stem, [])
        total_positive_boxes += len(predicted_boxes)
        if predicted_boxes:
            hit_any += 1
        gt_boxes = gt_by_stem[stem]
        best_iou = 0.0
        for gt_box in gt_boxes:
            for pred_box in predicted_boxes:
                best_iou = max(best_iou, bbox_iou(gt_box, pred_box))
        if best_iou >= 0.10:
            hit_iou_01 += 1
        if best_iou >= 0.30:
            hit_iou_03 += 1
        if best_iou >= 0.50:
            hit_iou_05 += 1

    for stem in negative_stems:
        predicted_boxes = predictions_by_stem.get(stem, [])
        total_negative_boxes += len(predicted_boxes)
        if predicted_boxes:
            false_positive_slices += 1

    def safe_div(numerator, denominator):
        return float(numerator) / float(denominator) if denominator else 0.0

    return {
        "num_positive_slices": positive_count,
        "num_negative_slices": negative_count,
        "slice_recall_any_box": safe_div(hit_any, positive_count),
        "slice_recall_iou_0.10": safe_div(hit_iou_01, positive_count),
        "slice_recall_iou_0.30": safe_div(hit_iou_03, positive_count),
        "slice_recall_iou_0.50": safe_div(hit_iou_05, positive_count),
        "background_false_positive_rate": safe_div(false_positive_slices, negative_count),
        "avg_boxes_per_positive_slice": safe_div(total_positive_boxes, positive_count),
        "avg_boxes_per_negative_slice": safe_div(total_negative_boxes, negative_count),
        "hit_any_count": hit_any,
        "hit_iou_0.10_count": hit_iou_01,
        "hit_iou_0.30_count": hit_iou_03,
        "hit_iou_0.50_count": hit_iou_05,
        "false_positive_slices": false_positive_slices,
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
            for box_xyxy in xyxy_values:
                boxes.append(yolo_xyxy_to_normalized_xywh(box_xyxy, image_width, image_height))
        predictions[stem] = boxes
    return predictions


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
        "background_false_positive_rate",
        "avg_boxes_per_positive_slice",
        "avg_boxes_per_negative_slice",
        "hit_any_count",
        "hit_iou_0.10_count",
        "hit_iou_0.30_count",
        "hit_iou_0.50_count",
        "false_positive_slices",
        "num_positive_slices",
        "num_negative_slices",
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
            item["slice_recall_any_box"],
            item["slice_recall_iou_0.30"],
            -item["background_false_positive_rate"],
            -item["avg_boxes_per_positive_slice"],
            -item["iou"],
            -item["conf"],
        ),
        reverse=True,
    )
    return sorted_results[0]


def choose_topk(results, top_k=5):
    if top_k <= 0:
        return []
    sorted_results = sorted(
        results,
        key=lambda item: (
            item["slice_recall_any_box"],
            item["slice_recall_iou_0.30"],
            -item["background_false_positive_rate"],
            -item["avg_boxes_per_positive_slice"],
            -item["iou"],
            -item["conf"],
        ),
        reverse=True,
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
            f"- Slice recall any box: `{_format_metric(recommendation['slice_recall_any_box'])}`",
            f"- Slice recall IoU@0.30: `{_format_metric(recommendation['slice_recall_iou_0.30'])}`",
            f"- Background false positive rate: `{_format_metric(recommendation['background_false_positive_rate'])}`",
            "",
        ])

    if shortlist:
        lines.extend([
            "## Top Candidates",
            "",
            "| Rank | IoU | Conf | Recall Any | Recall IoU@0.30 | BG FP Rate | Avg Boxes/Positive |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for index, item in enumerate(shortlist, start=1):
            lines.append(
                f"| {index} | "
                f"{_format_metric(item['iou'])} | "
                f"{_format_metric(item['conf'])} | "
                f"{_format_metric(item['slice_recall_any_box'])} | "
                f"{_format_metric(item['slice_recall_iou_0.30'])} | "
                f"{_format_metric(item['background_false_positive_rate'])} | "
                f"{_format_metric(item['avg_boxes_per_positive_slice'])} |"
            )
        lines.append("")

    lines.extend([
        "## Full Grid",
        "",
        "| IoU | Conf | Recall Any | Recall IoU@0.10 | Recall IoU@0.30 | Recall IoU@0.50 | BG FP Rate | Avg Boxes/Positive | Avg Boxes/Negative |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in payload["results"]:
        lines.append(
            f"| {_format_metric(item['iou'])} | "
            f"{_format_metric(item['conf'])} | "
            f"{_format_metric(item['slice_recall_any_box'])} | "
            f"{_format_metric(item['slice_recall_iou_0.10'])} | "
            f"{_format_metric(item['slice_recall_iou_0.30'])} | "
            f"{_format_metric(item['slice_recall_iou_0.50'])} | "
            f"{_format_metric(item['background_false_positive_rate'])} | "
            f"{_format_metric(item['avg_boxes_per_positive_slice'])} | "
            f"{_format_metric(item['avg_boxes_per_negative_slice'])} |"
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

    from ultralytics import YOLO

    model = YOLO(str(Path(args.model).resolve()))

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
            results.append(metrics)

    recommendation = choose_recommendation(results)
    recommended_topk = choose_topk(results)

    model_name = Path(args.model).stem
    out_dir = Path(args.out_dir) / f"{model_name}_{args.split}"
    payload = {
        "model": str(Path(args.model).resolve()),
        "data_yaml": str(dataset_info["data_yaml"]),
        "normalized_data_yaml": str(dataset_info["normalized_yaml"]),
        "dataset_root": str(dataset_info["dataset_root"]),
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
