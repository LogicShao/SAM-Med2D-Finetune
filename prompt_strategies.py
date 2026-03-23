from __future__ import annotations

from typing import Any

import cv2
import numpy as np


PREDICTION_CLASS_ORDER = ("WT", "TC", "ET")
CLASS_PROMPT_VARIANTS = ("baseline", "class_boxes", "class_boxes_points", "class_boxes_points_mask")
ET_PROMPT_VARIANTS = (
    "default",
    "q92_pad4_p1_n2",
    "q95_pad4_p1_n2",
    "q95_pad8_p1_n2_widefb",
    "q92_pad4_p2_n2",
    "q90_pad2_p1_n0",
)


def _resolve_et_prompt_variant(variant_name):
    presets = {
        "default": {
            "primary_quantile": 0.90,
            "secondary_quantile": 0.80,
            "box_padding_pixels": 0,
            "positive_points": 1,
            "negative_points": 4,
            "negative_region": "roi_ring",
            "fallback_mode": "shrunk_roi",
            "fallback_box_scale": 0.60,
            "min_component_pixels": 1,
            "fallback_on_small_candidate": False,
            "fragment_component_limit": None,
            "fragment_largest_ratio_min": 0.0,
            "fallback_on_fragmented_candidate": False,
        },
        "q92_pad4_p1_n2": {
            "primary_quantile": 0.92,
            "secondary_quantile": 0.85,
            "box_padding_pixels": 4,
            "positive_points": 1,
            "negative_points": 2,
            "negative_region": "tc_ring",
            "fallback_mode": "shrunk_roi",
            "fallback_box_scale": 0.60,
            "min_component_pixels": 9,
            "fallback_on_small_candidate": False,
            "fragment_component_limit": None,
            "fragment_largest_ratio_min": 0.0,
            "fallback_on_fragmented_candidate": False,
        },
        "q95_pad4_p1_n2": {
            "primary_quantile": 0.95,
            "secondary_quantile": 0.88,
            "box_padding_pixels": 4,
            "positive_points": 1,
            "negative_points": 2,
            "negative_region": "tc_ring",
            "fallback_mode": "shrunk_roi",
            "fallback_box_scale": 0.62,
            "min_component_pixels": 9,
            "fallback_on_small_candidate": True,
            "fragment_component_limit": None,
            "fragment_largest_ratio_min": 0.0,
            "fallback_on_fragmented_candidate": False,
        },
        "q95_pad8_p1_n2_widefb": {
            "primary_quantile": 0.95,
            "secondary_quantile": 0.90,
            "box_padding_pixels": 8,
            "positive_points": 1,
            "negative_points": 2,
            "negative_region": "tc_ring",
            "fallback_mode": "wide_roi",
            "fallback_box_scale": 0.72,
            "min_component_pixels": 16,
            "fallback_on_small_candidate": True,
            "fragment_component_limit": 3,
            "fragment_largest_ratio_min": 0.60,
            "fallback_on_fragmented_candidate": True,
        },
        "q92_pad4_p2_n2": {
            "primary_quantile": 0.92,
            "secondary_quantile": 0.85,
            "box_padding_pixels": 4,
            "positive_points": 2,
            "negative_points": 2,
            "negative_region": "tc_ring",
            "fallback_mode": "shrunk_roi",
            "fallback_box_scale": 0.60,
            "min_component_pixels": 9,
            "fallback_on_small_candidate": False,
            "fragment_component_limit": None,
            "fragment_largest_ratio_min": 0.0,
            "fallback_on_fragmented_candidate": False,
        },
        "q90_pad2_p1_n0": {
            "primary_quantile": 0.90,
            "secondary_quantile": 0.82,
            "box_padding_pixels": 2,
            "positive_points": 1,
            "negative_points": 0,
            "negative_region": "roi_ring",
            "fallback_mode": "shrunk_roi",
            "fallback_box_scale": 0.60,
            "min_component_pixels": 1,
            "fallback_on_small_candidate": False,
            "fragment_component_limit": None,
            "fragment_largest_ratio_min": 0.0,
            "fallback_on_fragmented_candidate": False,
        },
    }
    variant_name = str(variant_name)
    if variant_name not in presets:
        raise ValueError(f"Unsupported et_prompt_variant: {variant_name}. Expected one of {ET_PROMPT_VARIANTS}.")
    return {
        "name": variant_name,
        **presets[variant_name],
    }


def _clamp_box(box_xyxy, image_size):
    if box_xyxy is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    x1 = float(np.clip(x1, 0.0, image_size - 1.0))
    y1 = float(np.clip(y1, 0.0, image_size - 1.0))
    x2 = float(np.clip(x2, 0.0, image_size - 1.0))
    y2 = float(np.clip(y2, 0.0, image_size - 1.0))
    if x2 <= x1:
        x2 = min(float(image_size - 1.0), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(image_size - 1.0), y1 + 1.0)
    return [x1, y1, x2, y2]


def _scale_point_to_model(point_xy, image_size, original_width, original_height):
    if point_xy is None:
        return None
    x, y = [float(value) for value in point_xy]
    scale_x = float(image_size) / float(original_width)
    scale_y = float(image_size) / float(original_height)
    return [
        float(np.clip(x * scale_x, 0.0, image_size - 1.0)),
        float(np.clip(y * scale_y, 0.0, image_size - 1.0)),
    ]


def _binary_mask_to_box(mask_2d, image_size):
    y_indices, x_indices = np.where(np.asarray(mask_2d, dtype=np.uint8) > 0)
    if y_indices.size == 0:
        return None
    height, width = mask_2d.shape
    scale_x = float(image_size) / float(width)
    scale_y = float(image_size) / float(height)
    return _clamp_box(
        [
            float(x_indices.min() * scale_x),
            float(y_indices.min() * scale_y),
            float(x_indices.max() * scale_x),
            float(y_indices.max() * scale_y),
        ],
        image_size=image_size,
    )


def _largest_connected_component(mask_2d):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    if mask_uint8.ndim != 2 or not np.any(mask_uint8):
        return np.zeros_like(mask_uint8, dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask_uint8
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == component_index).astype(np.uint8)


def _shrink_binary_mask(mask_2d, kernel_size=5, iterations=1):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    if not np.any(mask_uint8):
        return np.zeros_like(mask_uint8, dtype=np.uint8)
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    shrunk = cv2.erode(mask_uint8, kernel, iterations=max(int(iterations), 1))
    if np.any(shrunk):
        return shrunk.astype(np.uint8)
    return mask_uint8


def _expand_binary_mask(mask_2d, kernel_size=5, iterations=1):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    if not np.any(mask_uint8):
        return np.zeros_like(mask_uint8, dtype=np.uint8)
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    return cv2.dilate(mask_uint8, kernel, iterations=max(int(iterations), 1)).astype(np.uint8)


def _distance_peak_point(mask_2d):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    if not np.any(mask_uint8):
        return None
    distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    max_index = int(np.argmax(distance))
    height, width = distance.shape
    y, x = divmod(max_index, width)
    return [float(x), float(y)]


def _brightest_point(mask_2d, intensity_2d):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    if not np.any(mask_uint8):
        return None
    masked_values = np.where(mask_uint8 > 0, np.asarray(intensity_2d, dtype=np.float32), -np.inf)
    max_index = int(np.argmax(masked_values))
    height, width = masked_values.shape
    y, x = divmod(max_index, width)
    return [float(x), float(y)]


def _sample_spread_points(mask_2d, max_points=4):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    ys, xs = np.where(mask_uint8 > 0)
    if ys.size == 0:
        return []
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    score_specs = (
        coords[:, 0] + coords[:, 1],
        coords[:, 0] - coords[:, 1],
        -coords[:, 0] + coords[:, 1],
        -coords[:, 0] - coords[:, 1],
    )
    selected = []
    seen = set()
    for score in score_specs:
        index = int(np.argmax(score))
        point = [float(coords[index, 0]), float(coords[index, 1])]
        key = (round(point[0], 2), round(point[1], 2))
        if key in seen:
            continue
        seen.add(key)
        selected.append(point)
        if len(selected) >= int(max_points):
            break
    if not selected:
        center_index = int(len(coords) // 2)
        selected.append([float(coords[center_index, 0]), float(coords[center_index, 1])])
    return selected


def _shrink_box(box_xyxy, image_size, scale_factor):
    if box_xyxy is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    half_width = max(width * float(scale_factor) / 2.0, 1.0)
    half_height = max(height * float(scale_factor) / 2.0, 1.0)
    return _clamp_box(
        [center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height],
        image_size=image_size,
    )


def _pad_box(box_xyxy, image_size, padding_pixels):
    if box_xyxy is None:
        return None
    padding_pixels = float(max(padding_pixels, 0))
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return _clamp_box(
        [x1 - padding_pixels, y1 - padding_pixels, x2 + padding_pixels, y2 + padding_pixels],
        image_size=image_size,
    )


def _build_mask_prompt(mask_2d, image_size):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    if mask_uint8.ndim != 2 or not np.any(mask_uint8):
        return None
    resized = cv2.resize(mask_uint8.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0.0, 1.0)
    return resized.astype(np.float32)


def _serialize_box_list(box_xyxy):
    if box_xyxy is None:
        return []
    return [list(box_xyxy)]


def _build_prompt_payload(
    *,
    class_name,
    slice_index,
    base_prompt_info,
    prompt_source,
    prompt_reason,
    box_xyxy=None,
    points=None,
    point_labels=None,
    mask_prompt=None,
    notes=None,
    diagnostics=None,
):
    points = points or []
    point_labels = point_labels or []
    notes = [str(item) for item in (notes or [])]
    return {
        "class_name": str(class_name),
        "slice_index": int(slice_index),
        "source": str(prompt_source),
        "reason": str(prompt_reason),
        "box_xyxy": None if box_xyxy is None else [float(value) for value in box_xyxy],
        "boxes_xyxy": _serialize_box_list(box_xyxy),
        "points_xy": [[float(x), float(y)] for x, y in points],
        "point_labels": [int(value) for value in point_labels],
        "mask_input": None if mask_prompt is None else np.asarray(mask_prompt, dtype=np.float32),
        "mask_input_present": bool(mask_prompt is not None),
        "notes": notes,
        "diagnostics": {str(key): value for key, value in (diagnostics or {}).items()},
        "base_source": str(base_prompt_info.get("source", "prompt_provider")),
        "base_selected_box_xyxy": (
            None
            if base_prompt_info.get("selected_box_xyxy") is None
            else [float(value) for value in base_prompt_info["selected_box_xyxy"]]
        ),
        "primary_score": (
            None if base_prompt_info.get("primary_score") is None else float(base_prompt_info.get("primary_score"))
        ),
    }


def _component_stats(mask_2d):
    mask_uint8 = np.asarray(mask_2d, dtype=np.uint8)
    if mask_uint8.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_uint8.shape}")
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    component_count = max(int(num_labels) - 1, 0)
    if component_count <= 0:
        return {
            "mask": np.zeros_like(mask_uint8, dtype=np.uint8),
            "component_count": 0,
            "largest_component_pixels": 0,
            "largest_component_ratio": 0.0,
            "total_pixels": 0,
        }
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int32)
    largest_index = 1 + int(np.argmax(areas))
    largest_pixels = int(areas.max())
    total_pixels = int(mask_uint8.sum())
    largest_component_ratio = float(largest_pixels / max(total_pixels, 1))
    return {
        "mask": (labels == largest_index).astype(np.uint8),
        "component_count": component_count,
        "largest_component_pixels": largest_pixels,
        "largest_component_ratio": largest_component_ratio,
        "total_pixels": total_pixels,
    }


def _build_et_positive_points(candidate_mask, intensity_2d, count):
    count = max(int(count), 0)
    if count <= 0:
        return []
    points = []
    seen = set()
    primary = _brightest_point(candidate_mask, intensity_2d)
    if primary is not None:
        points.append(primary)
        seen.add((round(primary[0], 2), round(primary[1], 2)))
    if len(points) >= count:
        return points[:count]
    for point in _sample_spread_points(candidate_mask, max_points=max(count * 2, 4)):
        key = (round(point[0], 2), round(point[1], 2))
        if key in seen:
            continue
        points.append(point)
        seen.add(key)
        if len(points) >= count:
            break
    return points[:count]


def _build_et_negative_mask(*, negative_region, roi_mask, tc_mask, candidate_mask):
    if negative_region == "tc_ring" and np.any(tc_mask):
        base_negative_roi = np.asarray(tc_mask, dtype=np.uint8)
    else:
        base_negative_roi = np.asarray(roi_mask, dtype=np.uint8)
    expanded_candidate = _expand_binary_mask(candidate_mask, kernel_size=7, iterations=1)
    return np.logical_and(base_negative_roi > 0, expanded_candidate == 0).astype(np.uint8)


def build_class_specific_prompt_info(
    *,
    class_name,
    slice_index,
    brats_case,
    image_size,
    base_prompt_info,
    predicted_masks,
    class_prompt_variant,
    et_prompt_variant="default",
):
    variant = str(class_prompt_variant)
    if variant not in CLASS_PROMPT_VARIANTS:
        raise ValueError(
            f"Unsupported class_prompt_variant: {variant}. Expected one of {CLASS_PROMPT_VARIANTS}."
        )

    base_box = base_prompt_info.get("selected_box_xyxy")
    if variant == "baseline" or class_name == "WT":
        return _build_prompt_payload(
            class_name=class_name,
            slice_index=slice_index,
            base_prompt_info=base_prompt_info,
            prompt_source="yolo_box" if base_box is not None else "skip_no_box",
            prompt_reason="baseline_prompt" if base_box is not None else "baseline_no_box",
            box_xyxy=base_box,
        )

    use_points = variant in {"class_boxes_points", "class_boxes_points_mask"}
    use_mask_input = variant == "class_boxes_points_mask"
    height, width = brats_case.shape[:2]

    if class_name == "TC":
        wt_mask = np.asarray(predicted_masks.get("WT", np.zeros((height, width), dtype=np.uint8)), dtype=np.uint8)
        if np.any(wt_mask):
            core_mask = _largest_connected_component(_shrink_binary_mask(wt_mask, kernel_size=5, iterations=1))
            edge_ring = np.logical_and(wt_mask > 0, _shrink_binary_mask(wt_mask, kernel_size=9, iterations=1) == 0)
            box_xyxy = _binary_mask_to_box(core_mask, image_size=image_size)
            positive = _distance_peak_point(core_mask)
            negatives = _sample_spread_points(edge_ring.astype(np.uint8), max_points=4) if use_points else []
            points_xy = []
            point_labels = []
            if use_points and positive is not None:
                points_xy.append(_scale_point_to_model(positive, image_size, width, height))
                point_labels.append(1)
                for point in negatives:
                    points_xy.append(_scale_point_to_model(point, image_size, width, height))
                    point_labels.append(0)
            return _build_prompt_payload(
                class_name=class_name,
                slice_index=slice_index,
                base_prompt_info=base_prompt_info,
                prompt_source="wt_core_roi",
                prompt_reason="eroded_wt_mask",
                box_xyxy=box_xyxy,
                points=points_xy,
                point_labels=point_labels,
                mask_prompt=_build_mask_prompt(core_mask, image_size) if use_mask_input else None,
                notes=[] if np.any(edge_ring) else ["tc_negative_ring_empty"],
            )

        fallback_box = _shrink_box(base_box, image_size=image_size, scale_factor=0.78)
        fallback_points = []
        fallback_labels = []
        if use_points and fallback_box is not None:
            x1, y1, x2, y2 = fallback_box
            fallback_points = [[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]
            fallback_labels = [1]
        return _build_prompt_payload(
            class_name=class_name,
            slice_index=slice_index,
            base_prompt_info=base_prompt_info,
            prompt_source="tc_fallback_from_yolo",
            prompt_reason="wt_mask_missing",
            box_xyxy=fallback_box,
            points=fallback_points,
            point_labels=fallback_labels,
            mask_prompt=None,
            notes=["tc_used_fallback_box"],
        )

    tc_mask = np.asarray(predicted_masks.get("TC", np.zeros((height, width), dtype=np.uint8)), dtype=np.uint8)
    wt_mask = np.asarray(predicted_masks.get("WT", np.zeros((height, width), dtype=np.uint8)), dtype=np.uint8)
    roi_mask = tc_mask if np.any(tc_mask) else wt_mask
    t1ce_slice = np.asarray(brats_case.normalized_volumes["t1ce"][:, :, int(slice_index)], dtype=np.float32)
    et_config = _resolve_et_prompt_variant(et_prompt_variant)
    et_notes = [f"et_variant:{et_config['name']}"]
    et_diagnostics = {
        "et_variant": et_config["name"],
        "roi_source": "TC" if np.any(tc_mask) else ("WT" if np.any(wt_mask) else "none"),
    }
    candidate_stage = "no_roi"

    if np.any(roi_mask):
        roi_values = t1ce_slice[roi_mask > 0]
        candidate = None
        candidate_stage = "none"
        for stage_name, quantile in (
            ("primary", float(et_config["primary_quantile"])),
            ("secondary", float(et_config["secondary_quantile"])),
        ):
            if stage_name == "secondary" and quantile <= 0.0:
                continue
            threshold = float(np.quantile(roi_values, quantile))
            threshold_mask = np.logical_and(roi_mask > 0, t1ce_slice >= threshold).astype(np.uint8)
            stats = _component_stats(threshold_mask)
            et_diagnostics[f"{stage_name}_quantile"] = quantile
            et_diagnostics[f"{stage_name}_threshold"] = threshold
            et_diagnostics[f"{stage_name}_candidate_pixels"] = stats["total_pixels"]
            et_diagnostics[f"{stage_name}_component_count"] = stats["component_count"]
            et_diagnostics[f"{stage_name}_largest_component_pixels"] = stats["largest_component_pixels"]
            et_diagnostics[f"{stage_name}_largest_component_ratio"] = stats["largest_component_ratio"]
            if stats["total_pixels"] <= 0:
                et_notes.append(f"et_{stage_name}_candidate_empty")
                continue
            if (
                et_config["fragment_component_limit"] is not None
                and stats["component_count"] > int(et_config["fragment_component_limit"])
                and stats["largest_component_ratio"] < float(et_config["fragment_largest_ratio_min"])
            ):
                et_notes.append("et_candidate_too_fragmented")
                et_diagnostics["fragmented_component_count"] = stats["component_count"]
                et_diagnostics["fragmented_largest_component_ratio"] = stats["largest_component_ratio"]
                if bool(et_config["fallback_on_fragmented_candidate"]):
                    et_notes.append("et_fallback_from_fragmented_candidate")
                    candidate = None
                    candidate_stage = "fragmented_fallback"
                    break
            candidate = stats["mask"]
            candidate_stage = stage_name
            if (
                stats["largest_component_pixels"] < int(et_config["min_component_pixels"])
                and bool(et_config["fallback_on_small_candidate"])
            ):
                et_notes.append("et_candidate_too_small")
                et_diagnostics["small_candidate_pixels"] = stats["largest_component_pixels"]
                et_notes.append("et_fallback_from_small_candidate")
                candidate = None
                candidate_stage = "small_fallback"
            break
        if np.any(candidate):
            negative_mask = _build_et_negative_mask(
                negative_region=str(et_config["negative_region"]),
                roi_mask=roi_mask,
                tc_mask=tc_mask,
                candidate_mask=candidate,
            )
            box_mask = candidate
            if int(et_config["box_padding_pixels"]) > 0:
                box_xyxy = _pad_box(
                    _binary_mask_to_box(box_mask, image_size=image_size),
                    image_size=image_size,
                    padding_pixels=int(et_config["box_padding_pixels"]),
                )
            else:
                box_xyxy = _binary_mask_to_box(box_mask, image_size=image_size)
            positive_points = _build_et_positive_points(candidate, t1ce_slice, count=et_config["positive_points"])
            negative_points = (
                _sample_spread_points(negative_mask.astype(np.uint8), max_points=int(et_config["negative_points"]))
                if use_points and int(et_config["negative_points"]) > 0
                else []
            )
            points_xy = []
            point_labels = []
            if use_points:
                for point in positive_points:
                    points_xy.append(_scale_point_to_model(point, image_size, width, height))
                    point_labels.append(1)
                for point in negative_points:
                    points_xy.append(_scale_point_to_model(point, image_size, width, height))
                    point_labels.append(0)
            et_diagnostics["selected_stage"] = candidate_stage
            et_diagnostics["selected_candidate_pixels"] = int(np.count_nonzero(candidate))
            et_diagnostics["selected_positive_points"] = len(positive_points)
            et_diagnostics["selected_negative_points"] = len(negative_points)
            return _build_prompt_payload(
                class_name=class_name,
                slice_index=slice_index,
                base_prompt_info=base_prompt_info,
                prompt_source="t1ce_bright_core",
                prompt_reason=f"roi_high_quantile_component_{candidate_stage}",
                box_xyxy=box_xyxy,
                points=points_xy,
                point_labels=point_labels,
                mask_prompt=_build_mask_prompt(candidate, image_size) if use_mask_input else None,
                notes=et_notes + ([] if np.any(negative_mask) else ["et_negative_ring_empty"]),
                diagnostics=et_diagnostics,
            )

    et_notes.append("et_used_fallback_box")
    fallback_mask = _largest_connected_component(_shrink_binary_mask(roi_mask, kernel_size=7, iterations=1))
    fallback_box = _binary_mask_to_box(fallback_mask, image_size=image_size)
    if fallback_box is None:
        fallback_box = _shrink_box(base_box, image_size=image_size, scale_factor=float(et_config["fallback_box_scale"]))
    elif et_config["fallback_mode"] == "wide_roi":
        roi_box = _binary_mask_to_box(roi_mask, image_size=image_size)
        fallback_box = _pad_box(
            roi_box if roi_box is not None else fallback_box,
            image_size=image_size,
            padding_pixels=int(et_config["box_padding_pixels"]),
        )
    else:
        fallback_box = _pad_box(
            fallback_box,
            image_size=image_size,
            padding_pixels=int(et_config["box_padding_pixels"]),
        )
    if fallback_box is None:
        fallback_box = _shrink_box(base_box, image_size=image_size, scale_factor=float(et_config["fallback_box_scale"]))
    fallback_points = []
    fallback_labels = []
    if use_points and fallback_box is not None and int(et_config["positive_points"]) > 0:
        if np.any(fallback_mask):
            positive_points = _build_et_positive_points(fallback_mask, t1ce_slice, count=et_config["positive_points"])
        else:
            x1, y1, x2, y2 = fallback_box
            positive_points = [[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]
        for point in positive_points:
            if np.any(fallback_mask):
                fallback_points.append(_scale_point_to_model(point, image_size, width, height))
            else:
                fallback_points.append([float(point[0]), float(point[1])])
            fallback_labels.append(1)
    et_diagnostics["selected_stage"] = candidate_stage if np.any(roi_mask) else "roi_missing"
    et_diagnostics["fallback_box_available"] = bool(fallback_box is not None)
    et_diagnostics["fallback_mask_pixels"] = int(np.count_nonzero(fallback_mask))
    return _build_prompt_payload(
        class_name=class_name,
        slice_index=slice_index,
        base_prompt_info=base_prompt_info,
        prompt_source="et_fallback_from_tc",
        prompt_reason="bright_component_missing",
        box_xyxy=fallback_box,
        points=fallback_points,
        point_labels=fallback_labels,
        mask_prompt=_build_mask_prompt(fallback_mask, image_size) if use_mask_input and np.any(fallback_mask) else None,
        notes=et_notes,
        diagnostics=et_diagnostics,
    )


def summarize_prompt_records(prompt_records):
    summary = {
        "num_records": len(prompt_records),
        "per_class": {},
    }
    for class_name in PREDICTION_CLASS_ORDER:
        summary["per_class"][class_name] = {
            "records": 0,
            "with_box": 0,
            "with_points": 0,
            "with_mask_input": 0,
            "source_counts": {},
            "reason_counts": {},
            "note_counts": {},
        }

    for record in prompt_records:
        class_name = str(record["class_name"])
        bucket = summary["per_class"][class_name]
        bucket["records"] += 1
        if record.get("box_xyxy") is not None:
            bucket["with_box"] += 1
        if record.get("points_xy"):
            bucket["with_points"] += 1
        if record.get("mask_input_present"):
            bucket["with_mask_input"] += 1
        source = str(record.get("source", "unknown"))
        reason = str(record.get("reason", "unknown"))
        bucket["source_counts"][source] = bucket["source_counts"].get(source, 0) + 1
        bucket["reason_counts"][reason] = bucket["reason_counts"].get(reason, 0) + 1
        for note in record.get("notes", []) or []:
            note = str(note)
            bucket["note_counts"][note] = bucket["note_counts"].get(note, 0) + 1
    return summary


def analyze_class_volume_consistency(class_volumes):
    wt = np.asarray(class_volumes["WT"], dtype=np.uint8)
    tc = np.asarray(class_volumes["TC"], dtype=np.uint8)
    et = np.asarray(class_volumes["ET"], dtype=np.uint8)
    voxel_counts = {
        "ET": int(np.count_nonzero(et)),
        "TC": int(np.count_nonzero(tc)),
        "WT": int(np.count_nonzero(wt)),
    }
    pairwise_equal = {
        "WT_TC": bool(np.array_equal(wt, tc)),
        "TC_ET": bool(np.array_equal(tc, et)),
        "WT_ET": bool(np.array_equal(wt, et)),
    }
    hierarchy_order_valid = bool(
        voxel_counts["ET"] <= voxel_counts["TC"] <= voxel_counts["WT"]
    )
    warnings = []
    if all(pairwise_equal.values()):
        warnings.append("WT/TC/ET masks are identical")
    if not hierarchy_order_valid:
        warnings.append("voxel order ET <= TC <= WT violated")
    return {
        "voxel_counts": voxel_counts,
        "pairwise_equal": pairwise_equal,
        "all_equal": bool(all(pairwise_equal.values())),
        "hierarchy_order_valid": hierarchy_order_valid,
        "warnings": warnings,
    }


def sanitize_prompt_records_for_json(prompt_records):
    sanitized = []
    for record in prompt_records:
        sanitized.append({
            **record,
            "mask_input": None,
        })
    return sanitized


def summarize_consistency_across_cases(results, key):
    total_cases = len(results)
    if total_cases <= 0:
        return {
            "num_cases": 0,
            "all_equal_cases": 0,
            "all_equal_ratio": 0.0,
            "hierarchy_order_valid_cases": 0,
            "hierarchy_order_valid_ratio": 0.0,
        }
    all_equal_cases = 0
    hierarchy_order_valid_cases = 0
    for item in results:
        checks = (item.get(key) or {})
        all_equal_cases += int(bool(checks.get("all_equal")))
        hierarchy_order_valid_cases += int(bool(checks.get("hierarchy_order_valid")))
    return {
        "num_cases": total_cases,
        "all_equal_cases": int(all_equal_cases),
        "all_equal_ratio": float(all_equal_cases / total_cases),
        "hierarchy_order_valid_cases": int(hierarchy_order_valid_cases),
        "hierarchy_order_valid_ratio": float(hierarchy_order_valid_cases / total_cases),
    }


def merge_prompt_source_counts(results):
    merged = {
        class_name: {}
        for class_name in PREDICTION_CLASS_ORDER
    }
    for item in results:
        prompt_summary = item.get("prompt_summary") or {}
        per_class = prompt_summary.get("per_class") or {}
        for class_name in PREDICTION_CLASS_ORDER:
            for source, count in (per_class.get(class_name, {}).get("source_counts") or {}).items():
                merged[class_name][source] = merged[class_name].get(source, 0) + int(count)
    return merged


def format_case_level_consistency(case_id, consistency):
    return {
        "case_id": str(case_id),
        "all_equal": bool(consistency.get("all_equal")),
        "hierarchy_order_valid": bool(consistency.get("hierarchy_order_valid")),
        "voxel_counts": consistency.get("voxel_counts") or {},
        "pairwise_equal": consistency.get("pairwise_equal") or {},
        "warnings": list(consistency.get("warnings") or []),
    }
