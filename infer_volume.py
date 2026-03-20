import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

from brats_case import BraTSCase
from model_factory import load_multitask_model
from postprocess_3d import postprocess_brats_masks


CLASS_NAMES = ("ET", "TC", "WT")
DEFAULT_YOLO_CHECKPOINT = "workdir_yolo/brats_yolo_dev_img320_v8m/weights/best.pt"
DEFAULT_YOLO_IMGSZ = 320
PROMPT_BOX_STRATEGIES = ("top1", "top2_merge")


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
    for class_name in CLASS_NAMES:
        strategy = class_overrides[class_name] or prompt_box_strategy
        strategy = str(strategy)
        if strategy not in PROMPT_BOX_STRATEGIES:
            raise ValueError(
                f"Unsupported prompt_box_strategy for {class_name}: {strategy}. "
                f"Expected one of {PROMPT_BOX_STRATEGIES}."
            )
        normalized[class_name] = strategy
    return normalized


class FullImageBoxPromptProvider:
    def __init__(self, image_size):
        self.image_size = image_size

    def should_skip_slice(self, slice_index, brats_case):
        del slice_index, brats_case
        return False

    def get_boxes(self, class_index, slice_index, brats_case):
        del class_index, slice_index, brats_case
        return torch.tensor(
            [[[0.0, 0.0, float(self.image_size - 1), float(self.image_size - 1)]]],
            dtype=torch.float32,
        )


class UpperBoundPromptProvider:
    def __init__(self, image_size):
        self.image_size = image_size

    def should_skip_slice(self, slice_index, brats_case):
        if not brats_case.has_segmentation():
            raise ValueError("prompt_mode=upper_bound requires '*_seg.nii.gz' to be present in the case directory.")
        return not brats_case.slice_has_any_gt(slice_index)

    def get_boxes(self, class_index, slice_index, brats_case):
        if not brats_case.has_segmentation():
            raise ValueError("prompt_mode=upper_bound requires '*_seg.nii.gz' to be present in the case directory.")

        class_name = CLASS_NAMES[class_index]
        gt_box = brats_case.get_gt_box(class_name, slice_index, image_size=self.image_size)
        if gt_box is None:
            return None
        return torch.tensor([[gt_box]], dtype=torch.float32)


def _configure_ultralytics_env(config_dir=".ultralytics"):
    config_dir = Path(config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLOv8_DIR", str(config_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(config_dir))
    return config_dir


class YoloBoxPromptProvider:
    def __init__(
        self,
        image_size,
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
        device,
    ):
        self.image_size = image_size
        self.yolo_checkpoint = Path(yolo_checkpoint).resolve()
        self.yolo_conf = float(yolo_conf)
        self.yolo_iou = float(yolo_iou)
        self.yolo_max_det = max(int(yolo_max_det), 1)
        self.yolo_topk = max(int(yolo_topk), 1)
        self.prompt_box_strategy = str(prompt_box_strategy)
        self.class_prompt_strategies = normalize_class_prompt_strategies(
            prompt_box_strategy=prompt_box_strategy,
            prompt_box_strategy_et=prompt_box_strategy_et,
            prompt_box_strategy_tc=prompt_box_strategy_tc,
            prompt_box_strategy_wt=prompt_box_strategy_wt,
        )
        self.top2_score_ratio = float(top2_score_ratio)
        self.top2_area_ratio_min = float(top2_area_ratio_min)
        self.top2_area_ratio_max = float(top2_area_ratio_max)
        self.top2_iou_max = float(top2_iou_max)
        self.device = self._normalize_device(device)
        self._candidate_cache = {}
        self._slice_cache = {}

        if not self.yolo_checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {self.yolo_checkpoint}")

        _configure_ultralytics_env()
        from ultralytics import YOLO

        self.model = YOLO(str(self.yolo_checkpoint))

    @staticmethod
    def _normalize_device(device):
        device = str(device)
        if device == "cuda":
            return "0"
        if device.startswith("cuda:"):
            return device.split(":", 1)[1]
        return device

    @staticmethod
    def _box_area_xyxy(box):
        x1, y1, x2, y2 = [float(value) for value in box]
        return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)

    @staticmethod
    def _box_iou_xyxy(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
        bx1, by1, bx2, by2 = [float(value) for value in box_b]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(inter_x2 - inter_x1, 0.0)
        inter_h = max(inter_y2 - inter_y1, 0.0)
        inter_area = inter_w * inter_h
        area_a = YoloBoxPromptProvider._box_area_xyxy(box_a)
        area_b = YoloBoxPromptProvider._box_area_xyxy(box_b)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    @staticmethod
    def _merge_boxes_xyxy(box_a, box_b):
        ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
        bx1, by1, bx2, by2 = [float(value) for value in box_b]
        return [
            min(ax1, bx1),
            min(ay1, by1),
            max(ax2, bx2),
            max(ay2, by2),
        ]

    def _effective_topk(self):
        if all(strategy == "top1" for strategy in self.class_prompt_strategies.values()):
            return 1
        return max(self.yolo_topk, 2)

    def _effective_max_det(self):
        return max(self.yolo_max_det, self._effective_topk())

    def _predict_candidates(self, slice_index, brats_case):
        cache_key = (brats_case.case_id, int(slice_index))
        if cache_key in self._candidate_cache:
            return self._candidate_cache[cache_key]

        pseudo_rgb = brats_case.get_pseudo_rgb_slice(slice_index)
        result = self.model.predict(
            source=pseudo_rgb,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            imgsz=DEFAULT_YOLO_IMGSZ,
            device=self.device,
            max_det=self._effective_max_det(),
            save=False,
            verbose=False,
        )[0]

        candidates = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            order = np.argsort(-confs)
            height, width = brats_case.shape[:2]
            scale_x = float(self.image_size) / float(width)
            scale_y = float(self.image_size) / float(height)
            for index in order.tolist():
                x1, y1, x2, y2 = xyxy[index].tolist()
                scaled_box = [
                    float(np.clip(x1 * scale_x, 0.0, self.image_size - 1.0)),
                    float(np.clip(y1 * scale_y, 0.0, self.image_size - 1.0)),
                    float(np.clip(x2 * scale_x, 0.0, self.image_size - 1.0)),
                    float(np.clip(y2 * scale_y, 0.0, self.image_size - 1.0)),
                ]
                candidates.append({
                    "box": scaled_box,
                    "score": float(confs[index]),
                    "area": float(self._box_area_xyxy(scaled_box)),
                })

        self._candidate_cache[cache_key] = candidates
        return candidates

    def _build_top1_decision(self, candidates):
        if not candidates:
            return {
                "selected_box": None,
                "decision_type": "skip_no_box",
                "rejected_reasons": [],
                "candidates": [],
            }

        return {
            "selected_box": list(candidates[0]["box"]),
            "decision_type": "top1_only",
            "rejected_reasons": [],
            "candidates": candidates,
        }

    def _build_top2_merge_decision(self, candidates):
        if not candidates:
            return {
                "selected_box": None,
                "decision_type": "skip_no_box",
                "rejected_reasons": [],
                "candidates": [],
            }

        if len(candidates) < 2:
            return {
                "selected_box": list(candidates[0]["box"]),
                "decision_type": "top1_fallback_single_candidate",
                "rejected_reasons": ["second_box_missing"],
                "candidates": candidates,
            }

        primary = candidates[0]
        secondary = candidates[1]
        rejected_reasons = []

        area1 = float(primary["area"])
        area2 = float(secondary["area"])
        if area1 <= 0.0:
            rejected_reasons.append("primary_area_nonpositive")
        else:
            if float(secondary["score"]) < self.top2_score_ratio * float(primary["score"]):
                rejected_reasons.append("score_ratio")
            if area2 < self.top2_area_ratio_min * area1:
                rejected_reasons.append("area_ratio_min")
            if area2 > self.top2_area_ratio_max * area1:
                rejected_reasons.append("area_ratio_max")

        pair_iou = self._box_iou_xyxy(primary["box"], secondary["box"])
        if pair_iou >= self.top2_iou_max:
            rejected_reasons.append("box_iou_max")

        if rejected_reasons:
            return {
                "selected_box": list(primary["box"]),
                "decision_type": "top1_second_box_rejected",
                "rejected_reasons": rejected_reasons,
                "pair_iou": float(pair_iou),
                "candidates": candidates,
            }

        return {
            "selected_box": self._merge_boxes_xyxy(primary["box"], secondary["box"]),
            "decision_type": "top2_merged",
            "rejected_reasons": [],
            "pair_iou": float(pair_iou),
            "candidates": candidates,
        }

    def _get_slice_decision(self, class_name, slice_index, brats_case):
        cache_key = (brats_case.case_id, int(slice_index), str(class_name))
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        topk = self._effective_topk()
        candidates = self._predict_candidates(slice_index, brats_case)[:topk]
        strategy = self.class_prompt_strategies[str(class_name)]

        if strategy == "top2_merge":
            decision = self._build_top2_merge_decision(candidates)
        else:
            decision = self._build_top1_decision(candidates)

        decision.update({
            "slice_index": int(slice_index),
            "case_id": brats_case.case_id,
            "class_name": str(class_name),
            "num_candidates_considered": len(candidates),
            "prompt_box_strategy": strategy,
        })
        self._slice_cache[cache_key] = decision
        return decision

    def should_skip_slice(self, slice_index, brats_case):
        for class_name in CLASS_NAMES:
            decision = self._get_slice_decision(class_name, slice_index, brats_case)
            if decision["selected_box"] is not None:
                return False
        return True

    def get_boxes(self, class_index, slice_index, brats_case):
        class_name = CLASS_NAMES[int(class_index)]
        decision = self._get_slice_decision(class_name, slice_index, brats_case)
        box = decision["selected_box"]
        if box is None:
            return None
        return torch.tensor([[box]], dtype=torch.float32)

    def build_case_prompt_report(self, brats_case):
        total_slices = int(brats_case.shape[2])
        slice_decisions = []
        overall_summary = {
            "total_slices": total_slices,
            "total_slice_class_pairs": int(total_slices * len(CLASS_NAMES)),
            "slice_class_pairs_with_prompt": 0,
            "slice_class_pairs_skipped": 0,
            "slice_class_pairs_top1_only": 0,
            "slice_class_pairs_top1_fallback_single_candidate": 0,
            "slice_class_pairs_top1_second_box_rejected": 0,
            "slice_class_pairs_top2_merged": 0,
            "slice_class_pairs_second_box_available": 0,
            "rejection_reasons": {},
        }
        per_class_summary = {
            class_name: {
                "strategy": self.class_prompt_strategies[class_name],
                "slices_with_prompt": 0,
                "slices_skipped": 0,
                "slices_top1_only": 0,
                "slices_top1_fallback_single_candidate": 0,
                "slices_top1_second_box_rejected": 0,
                "slices_top2_merged": 0,
                "slices_second_box_available": 0,
                "rejection_reasons": {},
            }
            for class_name in CLASS_NAMES
        }

        for slice_index in range(total_slices):
            for class_name in CLASS_NAMES:
                decision = self._get_slice_decision(class_name, slice_index, brats_case)
                class_summary = per_class_summary[class_name]
                decision_type = str(decision["decision_type"])

                if decision["selected_box"] is None:
                    overall_summary["slice_class_pairs_skipped"] += 1
                    class_summary["slices_skipped"] += 1
                else:
                    overall_summary["slice_class_pairs_with_prompt"] += 1
                    class_summary["slices_with_prompt"] += 1

                if decision_type == "top1_only":
                    overall_summary["slice_class_pairs_top1_only"] += 1
                    class_summary["slices_top1_only"] += 1
                elif decision_type == "top1_fallback_single_candidate":
                    overall_summary["slice_class_pairs_top1_fallback_single_candidate"] += 1
                    class_summary["slices_top1_fallback_single_candidate"] += 1
                elif decision_type == "top1_second_box_rejected":
                    overall_summary["slice_class_pairs_top1_second_box_rejected"] += 1
                    class_summary["slices_top1_second_box_rejected"] += 1
                elif decision_type == "top2_merged":
                    overall_summary["slice_class_pairs_top2_merged"] += 1
                    class_summary["slices_top2_merged"] += 1

                if len(decision["candidates"]) >= 2:
                    overall_summary["slice_class_pairs_second_box_available"] += 1
                    class_summary["slices_second_box_available"] += 1

                for reason in decision.get("rejected_reasons", []):
                    overall_summary["rejection_reasons"][reason] = overall_summary["rejection_reasons"].get(reason, 0) + 1
                    class_summary["rejection_reasons"][reason] = class_summary["rejection_reasons"].get(reason, 0) + 1

                slice_decisions.append({
                    "slice_index": int(decision["slice_index"]),
                    "class_name": class_name,
                    "prompt_box_strategy": decision["prompt_box_strategy"],
                    "decision_type": decision_type,
                    "num_candidates_considered": int(decision["num_candidates_considered"]),
                    "candidate_scores": [float(item["score"]) for item in decision["candidates"]],
                    "candidate_areas": [float(item["area"]) for item in decision["candidates"]],
                    "candidate_boxes": [list(item["box"]) for item in decision["candidates"]],
                    "selected_box": list(decision["selected_box"]) if decision["selected_box"] is not None else None,
                    "pair_iou": float(decision["pair_iou"]) if "pair_iou" in decision else None,
                    "rejected_reasons": list(decision.get("rejected_reasons", [])),
                })

        return {
            "case_id": brats_case.case_id,
            "prompt_mode": "yolo_box",
            "class_strategies": self.class_prompt_strategies,
            "summary": overall_summary,
            "per_class_summary": per_class_summary,
            "slice_decisions": slice_decisions,
        }


PROMPT_PROVIDERS = {
    "full_image_box": FullImageBoxPromptProvider,
    "upper_bound": UpperBoundPromptProvider,
    "yolo_box": YoloBoxPromptProvider,
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_torch_device(device):
    device = str(device)
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def parse_args():
    parser = argparse.ArgumentParser(description="Whole-case inference for BraTS volumes.")
    parser.add_argument("--case_dir", required=True, help="BraTS case directory with 4 modality NIfTI files.")
    parser.add_argument("--output_dir", required=True, help="Directory for NIfTI outputs and case_meta.json.")
    parser.add_argument("--sam_checkpoint", required=True, help="Base SAM-Med2D checkpoint path.")
    parser.add_argument("--finetuned_checkpoint", required=True, help="Adapter .pth or LoRA adapter directory.")
    parser.add_argument("--finetune_method", required=True, choices=["adapter", "lora"])
    parser.add_argument("--prompt_mode", default="full_image_box", choices=sorted(PROMPT_PROVIDERS))
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
    parser.add_argument("--yolo_max_det", type=int, default=2)
    parser.add_argument("--yolo_topk", type=int, default=2)
    parser.add_argument("--prompt_box_strategy", default="top1", choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--prompt_box_strategy_et", default=None, choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--prompt_box_strategy_tc", default=None, choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--prompt_box_strategy_wt", default=None, choices=PROMPT_BOX_STRATEGIES)
    parser.add_argument("--top2_score_ratio", type=float, default=0.5)
    parser.add_argument("--top2_area_ratio_min", type=float, default=0.1)
    parser.add_argument("--top2_area_ratio_max", type=float, default=2.0)
    parser.add_argument("--top2_iou_max", type=float, default=0.9)
    parser.add_argument("--postprocess", type=str_to_bool, default=False)
    parser.add_argument("--closing_radius", type=int, default=1)
    parser.add_argument("--opening_radius", type=int, default=1)
    parser.add_argument("--wt_keep_largest", type=str_to_bool, default=True)
    parser.add_argument("--keep_topk_tc", type=int, default=2)
    parser.add_argument("--keep_topk_et", type=int, default=2)
    parser.add_argument("--z_smooth_iterations", type=int, default=1)
    return parser.parse_args()


def build_prompt_provider(
    prompt_mode,
    image_size,
    yolo_checkpoint=None,
    yolo_conf=0.05,
    yolo_iou=0.60,
    yolo_max_det=2,
    yolo_topk=2,
    prompt_box_strategy="top1",
    prompt_box_strategy_et=None,
    prompt_box_strategy_tc=None,
    prompt_box_strategy_wt=None,
    top2_score_ratio=0.5,
    top2_area_ratio_min=0.1,
    top2_area_ratio_max=2.0,
    top2_iou_max=0.9,
    device="cpu",
):
    provider_class = PROMPT_PROVIDERS[prompt_mode]
    if prompt_mode == "yolo_box":
        return provider_class(
            image_size=image_size,
            yolo_checkpoint=yolo_checkpoint,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_max_det=yolo_max_det,
            yolo_topk=yolo_topk,
            prompt_box_strategy=prompt_box_strategy,
            prompt_box_strategy_et=prompt_box_strategy_et,
            prompt_box_strategy_tc=prompt_box_strategy_tc,
            prompt_box_strategy_wt=prompt_box_strategy_wt,
            top2_score_ratio=top2_score_ratio,
            top2_area_ratio_min=top2_area_ratio_min,
            top2_area_ratio_max=top2_area_ratio_max,
            top2_iou_max=top2_iou_max,
            device=device,
        )
    return provider_class(image_size=image_size)


def run_volume_inference(model, brats_case, prompt_provider, image_size, threshold, device, use_amp):
    height, width, depth = brats_case.shape
    class_volumes = {
        class_name: np.zeros((height, width, depth), dtype=np.uint8)
        for class_name in CLASS_NAMES
    }

    amp_enabled = use_amp and device.type == "cuda"

    with torch.no_grad():
        for slice_index in tqdm(range(depth), desc=f"Infer {brats_case.case_id}"):
            if prompt_provider.should_skip_slice(slice_index=slice_index, brats_case=brats_case):
                continue

            slice_tensor = brats_case.get_slice_tensor(slice_index, image_size)
            input_tensor = torch.from_numpy(slice_tensor).unsqueeze(0).to(device=device, dtype=torch.float32)

            with autocast(device_type=device.type, enabled=amp_enabled):
                image_embeddings = model.image_encoder(input_tensor)
                dense_pe = model.prompt_encoder.get_dense_pe()

                for class_index, class_name in enumerate(CLASS_NAMES):
                    boxes = prompt_provider.get_boxes(
                        class_index=class_index,
                        slice_index=slice_index,
                        brats_case=brats_case,
                    )
                    if boxes is None:
                        continue
                    boxes = boxes.to(device=device, dtype=torch.float32)
                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                        points=None,
                        boxes=boxes,
                        masks=None,
                    )
                    low_res_masks, _ = model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    upscaled_masks = F.interpolate(
                        low_res_masks,
                        size=(image_size, image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    probability_map = torch.sigmoid(upscaled_masks)[0, 0].detach().cpu().numpy()

                    original_probability = cv2.resize(
                        probability_map,
                        (width, height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    class_volumes[class_name][:, :, slice_index] = (
                        original_probability >= threshold
                    ).astype(np.uint8)

    return class_volumes


def build_combined_label(class_volumes):
    combined = np.zeros_like(class_volumes["ET"], dtype=np.uint8)
    combined[class_volumes["WT"] > 0] = 2
    combined[class_volumes["TC"] > 0] = 1
    combined[class_volumes["ET"] > 0] = 4
    return combined


def _to_json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(output_path, payload):
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(_to_json_compatible(payload), indent=2),
        encoding="utf-8",
    )


def save_mask_outputs(brats_case, output_dir, class_volumes, combined_label, prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename_prefix = f"{prefix}_" if prefix else ""
    brats_case.save_nifti(class_volumes["ET"].astype(np.uint8), output_dir / f"{filename_prefix}ET.nii.gz")
    brats_case.save_nifti(class_volumes["TC"].astype(np.uint8), output_dir / f"{filename_prefix}TC.nii.gz")
    brats_case.save_nifti(class_volumes["WT"].astype(np.uint8), output_dir / f"{filename_prefix}WT.nii.gz")
    brats_case.save_nifti(combined_label.astype(np.uint8), output_dir / f"{filename_prefix}combined_label.nii.gz")


def save_case_meta(brats_case, output_dir, meta):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    brats_case.write_case_meta(output_dir, meta)


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
    }


def build_case_meta(
    brats_case,
    output_dir,
    prompt_mode,
    finetune_method,
    sam_checkpoint,
    finetuned_checkpoint,
    image_size,
    threshold,
    postprocess_config,
    postprocess_report_path=None,
    prompt_report_path=None,
    yolo_config=None,
):
    return {
        "case_id": brats_case.case_id,
        "case_dir": str(brats_case.case_dir.resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "shape": list(brats_case.shape),
        "affine": brats_case.affine.tolist(),
        "voxel_spacing": list(brats_case.header.get_zooms()[:3]),
        "class_order": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "combined_label_map": {
            "1": "NCR/NET (TC minus ET)",
            "2": "ED (WT minus TC)",
            "4": "ET",
        },
        "prompt_mode": prompt_mode,
        "finetune_method": finetune_method,
        "sam_checkpoint": str(Path(sam_checkpoint).resolve()),
        "finetuned_checkpoint": str(Path(finetuned_checkpoint).resolve()),
        "image_size": image_size,
        "threshold": threshold,
        "normalization": {
            "mode": "per_volume_minmax_nonzero",
            "modalities": brats_case.normalization_stats,
        },
        "modality_paths": {
            key: str(path.resolve()) for key, path in brats_case.modality_paths.items()
        },
        "segmentation_path": str(brats_case.segmentation_path.resolve()) if brats_case.segmentation_path else None,
        "postprocess": {
            **postprocess_config,
            "report_path": str(postprocess_report_path.resolve()) if postprocess_report_path else None,
        },
        "prompt_report_path": str(prompt_report_path.resolve()) if prompt_report_path else None,
        "yolo": yolo_config,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = resolve_torch_device(args.device)

    brats_case = BraTSCase.from_dir(args.case_dir)
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
        device=args.device,
    )
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

    class_volumes = run_volume_inference(
        model=model,
        brats_case=brats_case,
        prompt_provider=prompt_provider,
        image_size=args.image_size,
        threshold=args.threshold,
        device=device,
        use_amp=args.use_amp,
    )
    combined_label = build_combined_label(class_volumes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_mask_outputs(brats_case, output_dir, class_volumes, combined_label)

    saved_files = [
        "ET.nii.gz",
        "TC.nii.gz",
        "WT.nii.gz",
        "combined_label.nii.gz",
    ]

    postprocess_config = build_postprocess_config(
        enabled=args.postprocess,
        closing_radius=args.closing_radius,
        opening_radius=args.opening_radius,
        wt_keep_largest=args.wt_keep_largest,
        keep_topk_tc=args.keep_topk_tc,
        keep_topk_et=args.keep_topk_et,
        z_smooth_iterations=args.z_smooth_iterations,
    )

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
        post_combined_label = build_combined_label(postprocessed_volumes)
        save_mask_outputs(
            brats_case,
            output_dir,
            postprocessed_volumes,
            post_combined_label,
            prefix="post",
        )

        postprocess_report = {
            "case_id": brats_case.case_id,
            "output_dir": str(output_dir.resolve()),
            **postprocess_report,
        }
        postprocess_report_path = output_dir / "postprocess_report.json"
        save_json(postprocess_report_path, postprocess_report)
        saved_files.extend([
            "post_ET.nii.gz",
            "post_TC.nii.gz",
            "post_WT.nii.gz",
            "post_combined_label.nii.gz",
            "postprocess_report.json",
        ])

    prompt_report_path = None
    if hasattr(prompt_provider, "build_case_prompt_report"):
        prompt_report_path = output_dir / "prompt_stats.json"
        prompt_report = prompt_provider.build_case_prompt_report(brats_case)
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
        )
        save_json(prompt_report_path, prompt_report)
        saved_files.append("prompt_stats.json")

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
        ),
    )
    save_case_meta(brats_case, output_dir, meta)

    print(json.dumps({
        "case_id": brats_case.case_id,
        "output_dir": str(output_dir.resolve()),
        "saved_files": saved_files + ["case_meta.json"],
    }, indent=2))


if __name__ == "__main__":
    main()
