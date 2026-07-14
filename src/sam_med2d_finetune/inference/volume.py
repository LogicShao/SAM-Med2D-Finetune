from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.nn import functional as F
from tqdm import tqdm

from sam_med2d_finetune.brats.case import BraTSCase
from sam_med2d_finetune.brats.constants import BRATS_CLASS_NAMES as CLASS_NAMES, PREDICTION_CLASS_ORDER
from sam_med2d_finetune.inference.config import (
    DEFAULT_YOLO_CHECKPOINT,
    DEFAULT_YOLO_IMGSZ,
    PROMPT_BOX_STRATEGIES,
    Z_PROMPT_MODES,
    build_postprocess_config,
    build_yolo_prompt_config,
    normalize_class_prompt_strategies,
)
from sam_med2d_finetune.inference.io import (
    build_case_meta,
    build_combined_label,
    save_case_meta,
    save_json,
    save_mask_outputs,
)
from sam_med2d_finetune.inference.postprocess import postprocess_brats_masks
from sam_med2d_finetune.inference.prompts import (
    CLASS_PROMPT_VARIANTS,
    ET_PROMPT_VARIANTS,
    analyze_class_volume_consistency,
    build_class_specific_prompt_info,
    sanitize_prompt_records_for_json,
    summarize_prompt_records,
)
from sam_med2d_finetune.models.factory import load_multitask_model
from sam_med2d_finetune.utils.cli import resolve_torch_device, str_to_bool


LOGGER = logging.getLogger(__name__)


@dataclass
class WTContinuityState:
    prev_slice_index: Optional[int] = None
    prev_box_xyxy: Optional[list[float]] = None
    prev_score: Optional[float] = None
    prev_lowres_prompt: Optional[np.ndarray] = None
    prev_binary_area: Optional[float] = None

    def clear(self):
        self.prev_slice_index = None
        self.prev_box_xyxy = None
        self.prev_score = None
        self.prev_lowres_prompt = None
        self.prev_binary_area = None

    def is_ready_for(self, slice_index):
        return (
            self.prev_slice_index is not None
            and int(slice_index) == int(self.prev_slice_index) + 1
            and self.prev_box_xyxy is not None
            and self.prev_lowres_prompt is not None
            and float(self.prev_binary_area or 0.0) > 0.0
        )

    def update(self, slice_index, box_xyxy, score, lowres_prompt, binary_area):
        self.prev_slice_index = int(slice_index)
        self.prev_box_xyxy = [float(value) for value in box_xyxy] if box_xyxy is not None else None
        self.prev_score = float(score) if score is not None else None
        self.prev_lowres_prompt = None if lowres_prompt is None else np.asarray(lowres_prompt, dtype=np.float32)
        self.prev_binary_area = float(binary_area)


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
        self.z_prompt_mode = str(z_prompt_mode)
        self.z_smooth_window = max(int(z_smooth_window), 1)
        self.z_fill_gap_max = max(int(z_fill_gap_max), 1)
        self.z_center_shift_max = float(z_center_shift_max)
        self.z_area_ratio_min = float(z_area_ratio_min)
        self.z_area_ratio_max = float(z_area_ratio_max)
        self.wt_continuity_enabled = bool(wt_continuity_enabled)
        self.wt_continuity_score_thresh = float(wt_continuity_score_thresh)
        self.wt_continuity_center_shift_max = float(wt_continuity_center_shift_max)
        self.wt_continuity_area_ratio_min = float(wt_continuity_area_ratio_min)
        self.wt_continuity_area_ratio_max = float(wt_continuity_area_ratio_max)
        self.wt_continuity_mask_dilate_iters = max(int(wt_continuity_mask_dilate_iters), 0)
        self.wt_continuity_mask_blur_kernel = max(int(wt_continuity_mask_blur_kernel), 1)
        if self.wt_continuity_mask_blur_kernel % 2 == 0:
            self.wt_continuity_mask_blur_kernel += 1
        self.device = self._normalize_device(device)
        self._candidate_cache = {}
        self._base_slice_cache = {}
        self._slice_cache = {}
        self._z_case_cache = set()
        self._runtime_case_reports = {}

        if self.z_prompt_mode not in Z_PROMPT_MODES:
            raise ValueError(
                f"Unsupported z_prompt_mode: {self.z_prompt_mode}. "
                f"Expected one of {Z_PROMPT_MODES}."
            )

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

    @staticmethod
    def _box_center_size_xyxy(box):
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0
        return center_x, center_y, width, height

    def _box_from_center_size(self, center_x, center_y, width, height):
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        x1 = float(np.clip(center_x - width / 2.0, 0.0, self.image_size - 1.0))
        y1 = float(np.clip(center_y - height / 2.0, 0.0, self.image_size - 1.0))
        x2 = float(np.clip(center_x + width / 2.0, 0.0, self.image_size - 1.0))
        y2 = float(np.clip(center_y + height / 2.0, 0.0, self.image_size - 1.0))
        return [x1, y1, x2, y2]

    def _clone_decision(self, decision):
        return {
            **decision,
            "selected_box": list(decision["selected_box"]) if decision.get("selected_box") is not None else None,
            "candidates": [
                {
                    **candidate,
                    "box": list(candidate["box"]),
                }
                for candidate in decision.get("candidates", [])
            ],
            "rejected_reasons": list(decision.get("rejected_reasons", [])),
        }

    @staticmethod
    def _boxes_almost_equal(box_a, box_b, atol=1e-4):
        if box_a is None or box_b is None:
            return box_a is None and box_b is None
        return all(abs(float(a) - float(b)) <= float(atol) for a, b in zip(box_a, box_b))

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

    def _get_base_slice_decision(self, class_name, slice_index, brats_case):
        cache_key = (brats_case.case_id, int(slice_index), str(class_name))
        if cache_key in self._base_slice_cache:
            return self._base_slice_cache[cache_key]

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
        self._base_slice_cache[cache_key] = decision
        return decision

    def _build_smoothed_box(self, base_decisions, slice_index):
        source_boxes = []
        source_slices = []
        for neighbor_index in range(
            max(0, int(slice_index) - self.z_smooth_window),
            min(len(base_decisions), int(slice_index) + self.z_smooth_window + 1),
        ):
            neighbor_box = base_decisions[neighbor_index]["selected_box"]
            if neighbor_box is None:
                continue
            source_boxes.append(neighbor_box)
            source_slices.append(int(neighbor_index))

        if len(source_boxes) < 2:
            return None, source_slices

        centers_sizes = np.asarray([self._box_center_size_xyxy(box) for box in source_boxes], dtype=np.float32)
        center_x, center_y, width, height = np.mean(centers_sizes, axis=0).tolist()
        return self._box_from_center_size(center_x, center_y, width, height), source_slices

    def _find_neighbor_box(self, base_decisions, slice_index, direction):
        max_distance = self.z_fill_gap_max
        current = int(slice_index)
        for distance in range(1, max_distance + 1):
            neighbor_index = current + direction * distance
            if neighbor_index < 0 or neighbor_index >= len(base_decisions):
                break
            neighbor_box = base_decisions[neighbor_index]["selected_box"]
            if neighbor_box is not None:
                return neighbor_index, neighbor_box
        return None, None

    def _is_stable_transition_pair(self, box_a, box_b):
        ax, ay, aw, ah = self._box_center_size_xyxy(box_a)
        bx, by, bw, bh = self._box_center_size_xyxy(box_b)
        center_shift = float(np.hypot(ax - bx, ay - by))
        if center_shift > self.z_center_shift_max:
            return False, "center_shift_max"

        area_a = max(self._box_area_xyxy(box_a), 1e-6)
        area_b = max(self._box_area_xyxy(box_b), 1e-6)
        area_ratio = area_b / area_a
        if area_ratio < self.z_area_ratio_min:
            return False, "z_area_ratio_min"
        if area_ratio > self.z_area_ratio_max:
            return False, "z_area_ratio_max"

        width_ratio = max(bw, 1e-6) / max(aw, 1e-6)
        height_ratio = max(bh, 1e-6) / max(ah, 1e-6)
        if width_ratio < self.z_area_ratio_min or height_ratio < self.z_area_ratio_min:
            return False, "z_size_ratio_min"
        if width_ratio > self.z_area_ratio_max or height_ratio > self.z_area_ratio_max:
            return False, "z_size_ratio_max"
        return True, None

    def _build_interpolated_box(self, base_decisions, slice_index):
        prev_index, prev_box = self._find_neighbor_box(base_decisions, slice_index, direction=-1)
        next_index, next_box = self._find_neighbor_box(base_decisions, slice_index, direction=1)
        if prev_box is None or next_box is None:
            return None, [], "missing_bracketing_boxes"

        is_stable, reject_reason = self._is_stable_transition_pair(prev_box, next_box)
        if not is_stable:
            return None, [int(prev_index), int(next_index)], reject_reason

        prev_values = np.asarray(self._box_center_size_xyxy(prev_box), dtype=np.float32)
        next_values = np.asarray(self._box_center_size_xyxy(next_box), dtype=np.float32)
        interp_range = float(next_index - prev_index)
        if interp_range <= 0.0:
            return None, [int(prev_index), int(next_index)], "invalid_interp_range"

        alpha = float(slice_index - prev_index) / interp_range
        interp_values = (1.0 - alpha) * prev_values + alpha * next_values
        center_x, center_y, width, height = interp_values.tolist()
        return self._box_from_center_size(center_x, center_y, width, height), [int(prev_index), int(next_index)], None

    def _prepare_case_z_decisions(self, brats_case):
        case_key = str(brats_case.case_id)
        if case_key in self._z_case_cache:
            return

        total_slices = int(brats_case.shape[2])
        for class_name in CLASS_NAMES:
            base_decisions = [
                self._get_base_slice_decision(class_name, slice_index, brats_case)
                for slice_index in range(total_slices)
            ]
            for slice_index, base_decision in enumerate(base_decisions):
                final_decision = self._clone_decision(base_decision)
                final_decision["base_selected_box"] = (
                    list(base_decision["selected_box"]) if base_decision["selected_box"] is not None else None
                )
                final_decision["z_prompt_mode"] = self.z_prompt_mode
                final_decision["z_action"] = "none"
                final_decision["z_source_slices"] = [int(slice_index)] if base_decision["selected_box"] is not None else []
                final_decision["z_rejected_reasons"] = []

                if self.z_prompt_mode == "smooth" and base_decision["selected_box"] is not None:
                    smoothed_box, source_slices = self._build_smoothed_box(base_decisions, slice_index)
                    if smoothed_box is not None and not self._boxes_almost_equal(smoothed_box, base_decision["selected_box"]):
                        final_decision["selected_box"] = smoothed_box
                        final_decision["z_action"] = "smoothed"
                        final_decision["z_source_slices"] = source_slices
                elif self.z_prompt_mode == "interpolate" and base_decision["selected_box"] is None:
                    interp_box, source_slices, reject_reason = self._build_interpolated_box(base_decisions, slice_index)
                    if interp_box is not None:
                        final_decision["selected_box"] = interp_box
                        final_decision["z_action"] = "interpolated"
                        final_decision["z_source_slices"] = source_slices
                    elif reject_reason is not None:
                        final_decision["z_rejected_reasons"] = [str(reject_reason)]
                        final_decision["z_source_slices"] = source_slices

                cache_key = (brats_case.case_id, int(slice_index), str(class_name))
                self._slice_cache[cache_key] = final_decision

        self._z_case_cache.add(case_key)

    def _get_slice_decision(self, class_name, slice_index, brats_case):
        self._prepare_case_z_decisions(brats_case)
        cache_key = (brats_case.case_id, int(slice_index), str(class_name))
        return self._slice_cache[cache_key]

    def start_case_runtime(self, case_id):
        self._runtime_case_reports[str(case_id)] = {
            "wt_continuity_summary": {
                "enabled": bool(self.wt_continuity_enabled),
                "eligible_total": 0,
                "trigger_total": 0,
                "trigger_reasons": {},
            },
            "wt_continuity_events": [],
            "prompt_events": [],
        }

    def _ensure_case_runtime(self, case_id):
        case_key = str(case_id)
        if case_key not in self._runtime_case_reports:
            self.start_case_runtime(case_key)
        return self._runtime_case_reports[case_key]

    def record_wt_continuity_eligibility(self, case_id):
        runtime = self._ensure_case_runtime(case_id)
        runtime["wt_continuity_summary"]["eligible_total"] += 1

    def record_prompt_event(self, case_id, payload):
        runtime = self._ensure_case_runtime(case_id)
        runtime["prompt_events"].append(payload)

    def record_wt_continuity_trigger(
        self,
        case_id,
        slice_index,
        trigger_reasons,
        source,
        primary_box_xyxy,
        primary_score,
        used_box_xyxy,
        prev_slice_index,
        prev_box_xyxy,
        prev_score,
        prev_binary_area,
        baseline_binary,
        continuity_binary,
    ):
        runtime = self._ensure_case_runtime(case_id)
        summary = runtime["wt_continuity_summary"]
        summary["trigger_total"] += 1
        for reason in trigger_reasons:
            summary["trigger_reasons"][reason] = summary["trigger_reasons"].get(reason, 0) + 1

        runtime["wt_continuity_events"].append({
            "slice_index": int(slice_index),
            "class_name": "WT",
            "trigger_reasons": [str(reason) for reason in trigger_reasons],
            "source": str(source),
            "primary_box_xyxy": list(primary_box_xyxy) if primary_box_xyxy is not None else None,
            "primary_score": float(primary_score) if primary_score is not None else None,
            "used_box_xyxy": list(used_box_xyxy) if used_box_xyxy is not None else None,
            "prev_slice_index": int(prev_slice_index) if prev_slice_index is not None else None,
            "prev_box_xyxy": list(prev_box_xyxy) if prev_box_xyxy is not None else None,
            "prev_score": float(prev_score) if prev_score is not None else None,
            "prev_binary_area": float(prev_binary_area) if prev_binary_area is not None else None,
            "_baseline_binary": np.asarray(baseline_binary, dtype=np.uint8),
            "_continuity_binary": np.asarray(continuity_binary, dtype=np.uint8),
        })

    @staticmethod
    def _slice_binary_dice(pred_mask, gt_mask, epsilon=1e-7):
        pred_mask = np.asarray(pred_mask, dtype=np.uint8)
        gt_mask = np.asarray(gt_mask, dtype=np.uint8)
        pred_sum = float(pred_mask.sum())
        gt_sum = float(gt_mask.sum())
        if pred_sum + gt_sum <= float(epsilon):
            return 1.0
        intersection = float(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
        return (2.0 * intersection + float(epsilon)) / (pred_sum + gt_sum + float(epsilon))

    def should_skip_slice(self, slice_index, brats_case):
        for class_name in CLASS_NAMES:
            decision = self._get_slice_decision(class_name, slice_index, brats_case)
            if decision["selected_box"] is not None:
                return False
        return True

    def get_prompt_info(self, class_index, slice_index, brats_case):
        class_name = CLASS_NAMES[int(class_index)]
        decision = self._get_slice_decision(class_name, slice_index, brats_case)
        selected_box = decision["selected_box"]
        candidates = decision.get("candidates", [])
        primary = candidates[0] if candidates else None
        return {
            "boxes": torch.tensor([[selected_box]], dtype=torch.float32) if selected_box is not None else None,
            "primary_box_xyxy": list(primary["box"]) if primary is not None else None,
            "primary_score": float(primary["score"]) if primary is not None else None,
            "source": str(decision.get("decision_type", "skip_no_box")),
            "class_name": class_name,
            "slice_index": int(slice_index),
            "selected_box_xyxy": list(selected_box) if selected_box is not None else None,
        }

    def get_boxes(self, class_index, slice_index, brats_case):
        prompt_info = self.get_prompt_info(class_index, slice_index, brats_case)
        return prompt_info["boxes"]

    def build_case_prompt_report(self, brats_case, gt_masks=None):
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
            "slice_class_pairs_z_smoothed": 0,
            "slice_class_pairs_z_interpolated": 0,
            "rejection_reasons": {},
            "z_rejection_reasons": {},
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
                "slices_z_smoothed": 0,
                "slices_z_interpolated": 0,
                "rejection_reasons": {},
                "z_rejection_reasons": {},
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

                z_action = str(decision.get("z_action", "none"))
                if z_action == "smoothed":
                    overall_summary["slice_class_pairs_z_smoothed"] += 1
                    class_summary["slices_z_smoothed"] += 1
                elif z_action == "interpolated":
                    overall_summary["slice_class_pairs_z_interpolated"] += 1
                    class_summary["slices_z_interpolated"] += 1

                for reason in decision.get("rejected_reasons", []):
                    overall_summary["rejection_reasons"][reason] = overall_summary["rejection_reasons"].get(reason, 0) + 1
                    class_summary["rejection_reasons"][reason] = class_summary["rejection_reasons"].get(reason, 0) + 1
                for reason in decision.get("z_rejected_reasons", []):
                    overall_summary["z_rejection_reasons"][reason] = overall_summary["z_rejection_reasons"].get(reason, 0) + 1
                    class_summary["z_rejection_reasons"][reason] = class_summary["z_rejection_reasons"].get(reason, 0) + 1

                slice_decisions.append({
                    "slice_index": int(decision["slice_index"]),
                    "class_name": class_name,
                    "prompt_box_strategy": decision["prompt_box_strategy"],
                    "z_prompt_mode": decision.get("z_prompt_mode", "none"),
                    "z_action": z_action,
                    "z_source_slices": list(decision.get("z_source_slices", [])),
                    "decision_type": decision_type,
                    "num_candidates_considered": int(decision["num_candidates_considered"]),
                    "candidate_scores": [float(item["score"]) for item in decision["candidates"]],
                    "candidate_areas": [float(item["area"]) for item in decision["candidates"]],
                    "candidate_boxes": [list(item["box"]) for item in decision["candidates"]],
                    "base_selected_box": list(decision["base_selected_box"]) if decision.get("base_selected_box") is not None else None,
                    "selected_box": list(decision["selected_box"]) if decision["selected_box"] is not None else None,
                    "pair_iou": float(decision["pair_iou"]) if "pair_iou" in decision else None,
                    "rejected_reasons": list(decision.get("rejected_reasons", [])),
                    "z_rejected_reasons": list(decision.get("z_rejected_reasons", [])),
                })

        runtime = self._ensure_case_runtime(brats_case.case_id)
        wt_continuity_summary = {
            **runtime["wt_continuity_summary"],
            "rescue": 0,
            "neutral": 0,
            "harm": 0,
        }
        wt_continuity_events = []
        gt_wt = None if gt_masks is None else gt_masks.get("WT")
        for event in runtime["wt_continuity_events"]:
            payload = {
                key: value
                for key, value in event.items()
                if not key.startswith("_")
            }
            if gt_wt is not None:
                gt_slice = gt_wt[:, :, int(event["slice_index"])]
                baseline_dice = self._slice_binary_dice(event["_baseline_binary"], gt_slice)
                continuity_dice = self._slice_binary_dice(event["_continuity_binary"], gt_slice)
                if continuity_dice > baseline_dice + 1e-4:
                    outcome = "rescue"
                elif continuity_dice + 1e-4 < baseline_dice:
                    outcome = "harm"
                else:
                    outcome = "neutral"
                wt_continuity_summary[outcome] += 1
                payload.update({
                    "baseline_slice_dice": float(baseline_dice),
                    "continuity_slice_dice": float(continuity_dice),
                    "outcome": outcome,
                })
            wt_continuity_events.append(payload)

        return {
            "case_id": brats_case.case_id,
            "prompt_mode": "yolo_box",
            "class_strategies": self.class_prompt_strategies,
            "summary": overall_summary,
            "per_class_summary": per_class_summary,
            "slice_decisions": slice_decisions,
            "wt_continuity": {
                "summary": wt_continuity_summary,
                "events": wt_continuity_events,
            },
            "runtime_prompt_events": runtime["prompt_events"],
        }


PROMPT_PROVIDERS = {
    "full_image_box": FullImageBoxPromptProvider,
    "upper_bound": UpperBoundPromptProvider,
    "yolo_box": YoloBoxPromptProvider,
}


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
    parser.add_argument("--class_prompt_variant", default="baseline", choices=CLASS_PROMPT_VARIANTS)
    parser.add_argument("--et_prompt_variant", default="default", choices=ET_PROMPT_VARIANTS)
    parser.add_argument("--top2_score_ratio", type=float, default=0.5)
    parser.add_argument("--top2_area_ratio_min", type=float, default=0.1)
    parser.add_argument("--top2_area_ratio_max", type=float, default=2.0)
    parser.add_argument("--top2_iou_max", type=float, default=0.9)
    parser.add_argument("--z_prompt_mode", default="none", choices=Z_PROMPT_MODES)
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
    z_prompt_mode="none",
    z_smooth_window=1,
    z_fill_gap_max=1,
    z_center_shift_max=64.0,
    z_area_ratio_min=0.25,
    z_area_ratio_max=4.0,
    wt_continuity_enabled=False,
    wt_continuity_score_thresh=0.15,
    wt_continuity_center_shift_max=48.0,
    wt_continuity_area_ratio_min=0.5,
    wt_continuity_area_ratio_max=2.0,
    wt_continuity_mask_dilate_iters=1,
    wt_continuity_mask_blur_kernel=3,
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
            z_prompt_mode=z_prompt_mode,
            z_smooth_window=z_smooth_window,
            z_fill_gap_max=z_fill_gap_max,
            z_center_shift_max=z_center_shift_max,
            z_area_ratio_min=z_area_ratio_min,
            z_area_ratio_max=z_area_ratio_max,
            wt_continuity_enabled=wt_continuity_enabled,
            wt_continuity_score_thresh=wt_continuity_score_thresh,
            wt_continuity_center_shift_max=wt_continuity_center_shift_max,
            wt_continuity_area_ratio_min=wt_continuity_area_ratio_min,
            wt_continuity_area_ratio_max=wt_continuity_area_ratio_max,
            wt_continuity_mask_dilate_iters=wt_continuity_mask_dilate_iters,
            wt_continuity_mask_blur_kernel=wt_continuity_mask_blur_kernel,
            device=device,
        )
    return provider_class(image_size=image_size)


def _predict_mask_from_prompt(
    model,
    image_embeddings,
    dense_pe,
    boxes,
    point_coords,
    point_labels,
    mask_input,
    image_size,
    original_width,
    original_height,
    threshold,
):
    points = None
    if point_coords is not None and point_labels is not None:
        points = (point_coords, point_labels)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=mask_input,
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
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    binary_mask = (original_probability >= threshold).astype(np.uint8)
    return {
        "lowres_prompt": low_res_masks[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
        "binary_mask": binary_mask,
    }


def _build_wt_coarse_mask_prompt(lowres_prompt, dilate_iters, blur_kernel):
    if lowres_prompt is None:
        return None
    clipped_logits = np.clip(np.asarray(lowres_prompt, dtype=np.float32), -20.0, 20.0)
    probability = 1.0 / (1.0 + np.exp(-clipped_logits))
    if int(blur_kernel) > 1:
        probability = cv2.GaussianBlur(
            probability,
            (int(blur_kernel), int(blur_kernel)),
            sigmaX=0.0,
        )
    binary = (probability >= 0.5).astype(np.uint8)
    if int(dilate_iters) > 0:
        binary = cv2.dilate(
            binary,
            np.ones((3, 3), dtype=np.uint8),
            iterations=int(dilate_iters),
        )
    coarse_mask = np.maximum(probability, binary.astype(np.float32))
    coarse_mask = np.ascontiguousarray(coarse_mask[None, None, :, :], dtype=np.float32)
    return torch.from_numpy(coarse_mask)


def _evaluate_wt_continuity_gate(prompt_info, wt_state, prompt_provider):
    if not bool(getattr(prompt_provider, "wt_continuity_enabled", False)):
        return False, []
    if not wt_state.is_ready_for(prompt_info["slice_index"]):
        return False, []

    reasons = []
    current_box = prompt_info.get("primary_box_xyxy") or prompt_info.get("selected_box_xyxy")
    if prompt_info.get("boxes") is None:
        reasons.append("missing_box")
    else:
        primary_score = prompt_info.get("primary_score")
        if primary_score is not None and float(primary_score) < float(prompt_provider.wt_continuity_score_thresh):
            reasons.append("low_score")

    if current_box is not None and wt_state.prev_box_xyxy is not None:
        current_cx, current_cy, _, _ = YoloBoxPromptProvider._box_center_size_xyxy(current_box)
        prev_cx, prev_cy, _, _ = YoloBoxPromptProvider._box_center_size_xyxy(wt_state.prev_box_xyxy)
        center_shift = float(np.hypot(current_cx - prev_cx, current_cy - prev_cy))
        if center_shift > float(prompt_provider.wt_continuity_center_shift_max):
            reasons.append("center_jump")

        current_area = max(YoloBoxPromptProvider._box_area_xyxy(current_box), 1e-6)
        prev_area = max(YoloBoxPromptProvider._box_area_xyxy(wt_state.prev_box_xyxy), 1e-6)
        area_ratio = current_area / prev_area
        if (
            area_ratio < float(prompt_provider.wt_continuity_area_ratio_min)
            or area_ratio > float(prompt_provider.wt_continuity_area_ratio_max)
        ):
            reasons.append("area_jump")

    return True, reasons


def _materialize_prompt_tensors(prompt_payload, device, mask_input_size=None):
    box_xyxy = prompt_payload.get("box_xyxy")
    points_xy = prompt_payload.get("points_xy") or []
    point_labels = prompt_payload.get("point_labels") or []
    mask_input = prompt_payload.get("mask_input")

    boxes = None
    if box_xyxy is not None:
        boxes = torch.tensor([[box_xyxy]], dtype=torch.float32, device=device)

    point_coords_tensor = None
    point_labels_tensor = None
    if points_xy:
        point_coords_tensor = torch.tensor([points_xy], dtype=torch.float32, device=device)
        point_labels_tensor = torch.tensor([point_labels], dtype=torch.int64, device=device)

    mask_input_tensor = None
    if mask_input is not None:
        if mask_input_size is not None and tuple(mask_input.shape) != tuple(mask_input_size):
            target_width = int(mask_input_size[1])
            target_height = int(mask_input_size[0])
            mask_input = cv2.resize(
                np.asarray(mask_input, dtype=np.float32),
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        mask_input_tensor = torch.from_numpy(
            np.ascontiguousarray(mask_input[None, None, :, :], dtype=np.float32)
        ).to(device=device, dtype=torch.float32)

    return boxes, point_coords_tensor, point_labels_tensor, mask_input_tensor


def run_volume_inference(
    model,
    brats_case,
    prompt_provider,
    image_size,
    threshold,
    device,
    use_amp,
    class_prompt_variant="baseline",
    et_prompt_variant="default",
):
    height, width, depth = brats_case.shape
    class_volumes = {
        class_name: np.zeros((height, width, depth), dtype=np.uint8)
        for class_name in CLASS_NAMES
    }
    runtime_prompt_records = []

    amp_enabled = use_amp and device.type == "cuda"
    wt_state = WTContinuityState()
    if hasattr(prompt_provider, "start_case_runtime"):
        prompt_provider.start_case_runtime(brats_case.case_id)

    with torch.no_grad():
        for slice_index in tqdm(range(depth), desc=f"Infer {brats_case.case_id}"):
            base_prompt_infos = {}
            has_any_prompt = False
            for class_index, class_name in enumerate(CLASS_NAMES):
                if hasattr(prompt_provider, "get_prompt_info"):
                    base_prompt_info = prompt_provider.get_prompt_info(
                        class_index=class_index,
                        slice_index=slice_index,
                        brats_case=brats_case,
                    )
                else:
                    boxes = prompt_provider.get_boxes(
                        class_index=class_index,
                        slice_index=slice_index,
                        brats_case=brats_case,
                    )
                    selected_box = None
                    if boxes is not None:
                        selected_box = boxes.detach().cpu().reshape(-1, 4)[0].tolist()
                        has_any_prompt = True
                    base_prompt_info = {
                        "boxes": boxes,
                        "primary_box_xyxy": list(selected_box) if selected_box is not None else None,
                        "primary_score": None,
                        "source": "prompt_provider",
                        "class_name": class_name,
                        "slice_index": int(slice_index),
                        "selected_box_xyxy": list(selected_box) if selected_box is not None else None,
                    }
                if base_prompt_info["boxes"] is not None:
                    has_any_prompt = True
                base_prompt_infos[class_name] = base_prompt_info

            wt_eligible, wt_trigger_reasons = _evaluate_wt_continuity_gate(
                base_prompt_infos["WT"],
                wt_state,
                prompt_provider,
            )
            if wt_eligible and hasattr(prompt_provider, "record_wt_continuity_eligibility"):
                prompt_provider.record_wt_continuity_eligibility(brats_case.case_id)
            wt_should_trigger = wt_eligible and bool(wt_trigger_reasons)

            if not has_any_prompt and not wt_should_trigger:
                wt_state.clear()
                continue

            slice_tensor = brats_case.get_slice_tensor(slice_index, image_size)
            input_tensor = torch.from_numpy(slice_tensor).unsqueeze(0).to(device=device, dtype=torch.float32)

            with autocast(device_type=device.type, enabled=amp_enabled):
                image_embeddings = model.image_encoder(input_tensor)
                dense_pe = model.prompt_encoder.get_dense_pe()
                predicted_masks_for_slice = {}
                for class_name in PREDICTION_CLASS_ORDER:
                    base_prompt_info = base_prompt_infos[class_name]
                    prompt_payload = build_class_specific_prompt_info(
                        class_name=class_name,
                        slice_index=slice_index,
                        brats_case=brats_case,
                        image_size=image_size,
                        base_prompt_info=base_prompt_info,
                        predicted_masks=predicted_masks_for_slice,
                        class_prompt_variant=class_prompt_variant,
                        et_prompt_variant=et_prompt_variant,
                    )

                    if hasattr(prompt_provider, "record_prompt_event"):
                        prompt_provider.record_prompt_event(
                            brats_case.case_id,
                            {
                                key: value
                                for key, value in sanitize_prompt_records_for_json([prompt_payload])[0].items()
                                if key != "mask_input"
                            },
                        )
                    runtime_prompt_records.append(sanitize_prompt_records_for_json([prompt_payload])[0])

                    boxes, point_coords, point_labels, mask_input = _materialize_prompt_tensors(
                        prompt_payload,
                        device,
                        mask_input_size=getattr(model.prompt_encoder, "mask_input_size", None),
                    )

                    if class_name != "WT" or not bool(getattr(prompt_provider, "wt_continuity_enabled", False)):
                        if boxes is None and point_coords is None and mask_input is None:
                            predicted_masks_for_slice[class_name] = np.zeros((height, width), dtype=np.uint8)
                            continue
                        result = _predict_mask_from_prompt(
                            model=model,
                            image_embeddings=image_embeddings,
                            dense_pe=dense_pe,
                            boxes=boxes,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            mask_input=mask_input,
                            image_size=image_size,
                            original_width=width,
                            original_height=height,
                            threshold=threshold,
                        )
                        class_volumes[class_name][:, :, slice_index] = result["binary_mask"]
                        predicted_masks_for_slice[class_name] = result["binary_mask"]
                        continue

                    wt_trigger_source = None
                    wt_used_box_xyxy = prompt_payload.get("box_xyxy")
                    final_result = None
                    baseline_binary = np.zeros((height, width), dtype=np.uint8)

                    if boxes is None and point_coords is None and mask_input is None:
                        if not wt_should_trigger:
                            wt_state.clear()
                            predicted_masks_for_slice[class_name] = np.zeros((height, width), dtype=np.uint8)
                            continue
                    else:
                        baseline_result = _predict_mask_from_prompt(
                            model=model,
                            image_embeddings=image_embeddings,
                            dense_pe=dense_pe,
                            boxes=boxes,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            mask_input=mask_input,
                            image_size=image_size,
                            original_width=width,
                            original_height=height,
                            threshold=threshold,
                        )
                        baseline_binary = baseline_result["binary_mask"]

                    if wt_should_trigger:
                        continuity_boxes = boxes
                        continuity_points = point_coords
                        continuity_labels = point_labels
                        wt_trigger_source = "current_prompt_plus_coarse_mask"
                        if continuity_boxes is None and wt_state.prev_box_xyxy is not None:
                            continuity_boxes = torch.tensor([[wt_state.prev_box_xyxy]], dtype=torch.float32, device=device)
                            continuity_points = None
                            continuity_labels = None
                            wt_trigger_source = "prev_box_plus_coarse_mask"
                            wt_used_box_xyxy = list(wt_state.prev_box_xyxy)

                        continuity_mask = _build_wt_coarse_mask_prompt(
                            lowres_prompt=wt_state.prev_lowres_prompt,
                            dilate_iters=getattr(prompt_provider, "wt_continuity_mask_dilate_iters", 1),
                            blur_kernel=getattr(prompt_provider, "wt_continuity_mask_blur_kernel", 3),
                        )
                        if continuity_boxes is not None and continuity_mask is not None:
                            final_result = _predict_mask_from_prompt(
                                model=model,
                                image_embeddings=image_embeddings,
                                dense_pe=dense_pe,
                                boxes=continuity_boxes,
                                point_coords=continuity_points,
                                point_labels=continuity_labels,
                                mask_input=continuity_mask.to(device=device, dtype=torch.float32),
                                image_size=image_size,
                                original_width=width,
                                original_height=height,
                                threshold=threshold,
                            )
                            if hasattr(prompt_provider, "record_wt_continuity_trigger"):
                                prompt_provider.record_wt_continuity_trigger(
                                    case_id=brats_case.case_id,
                                    slice_index=slice_index,
                                    trigger_reasons=wt_trigger_reasons,
                                    source=wt_trigger_source,
                                    primary_box_xyxy=base_prompt_info.get("primary_box_xyxy"),
                                    primary_score=base_prompt_info.get("primary_score"),
                                    used_box_xyxy=wt_used_box_xyxy,
                                    prev_slice_index=wt_state.prev_slice_index,
                                    prev_box_xyxy=wt_state.prev_box_xyxy,
                                    prev_score=wt_state.prev_score,
                                    prev_binary_area=wt_state.prev_binary_area,
                                    baseline_binary=baseline_binary,
                                    continuity_binary=final_result["binary_mask"],
                                )

                    if final_result is None:
                        if boxes is None and point_coords is None and mask_input is None:
                            wt_state.clear()
                            predicted_masks_for_slice[class_name] = np.zeros((height, width), dtype=np.uint8)
                            continue
                        final_result = baseline_result

                    class_volumes[class_name][:, :, slice_index] = final_result["binary_mask"]
                    predicted_masks_for_slice[class_name] = final_result["binary_mask"]
                    final_binary_area = float(final_result["binary_mask"].sum())
                    if final_binary_area <= 0.0:
                        wt_state.clear()
                        continue

                    wt_state.update(
                        slice_index=slice_index,
                        box_xyxy=wt_used_box_xyxy,
                        score=base_prompt_info.get("primary_score"),
                        lowres_prompt=final_result["lowres_prompt"],
                        binary_area=final_binary_area,
                    )

    raw_consistency = analyze_class_volume_consistency(class_volumes)
    for warning in raw_consistency.get("warnings", []):
        LOGGER.warning("Case %s raw mask warning: %s", brats_case.case_id, warning)

    inference_report = {
        "class_prompt_variant": str(class_prompt_variant),
        "et_prompt_variant": str(et_prompt_variant),
        "prompt_records": runtime_prompt_records,
        "prompt_summary": summarize_prompt_records(runtime_prompt_records),
        "raw_consistency": raw_consistency,
    }
    return class_volumes, inference_report


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
    else:
        postprocessed_volumes = class_volumes

    post_consistency = analyze_class_volume_consistency(postprocessed_volumes)
    for warning in post_consistency.get("warnings", []):
        LOGGER.warning("Case %s post mask warning: %s", brats_case.case_id, warning)

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
            "post": post_consistency,
        }
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

    print(json.dumps({
        "case_id": brats_case.case_id,
        "output_dir": str(output_dir.resolve()),
        "saved_files": saved_files + ["case_meta.json"],
    }, indent=2))


if __name__ == "__main__":
    main()
