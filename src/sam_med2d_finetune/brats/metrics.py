"""Patient-level BraTS segmentation metrics on a shared native-volume grid."""

from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import ndimage

from sam_med2d_finetune.brats.constants import BRATS_CLASS_NAMES as CLASS_NAMES



def _as_mask(mask, name):
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D mask, got shape={array.shape}.")
    return array.astype(bool, copy=False)


def _validate_spacing(spacing_mm):
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != 3 or any(not np.isfinite(value) or value <= 0.0 for value in spacing):
        raise ValueError(f"spacing_mm must contain three positive values, got {spacing_mm}.")
    return spacing


def _surface(mask):
    if not np.any(mask):
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)


def compute_hd95_mm(pred_mask, gt_mask, spacing_mm):
    """Compute symmetric HD95 in millimetres, or ``None`` when a surface is absent."""
    pred = _as_mask(pred_mask, "pred_mask")
    gt = _as_mask(gt_mask, "gt_mask")
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: prediction {pred.shape}, ground truth {gt.shape}.")
    if not np.any(pred) or not np.any(gt):
        return None

    spacing = _validate_spacing(spacing_mm)
    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    distances_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    distances_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    surface_distances = np.concatenate((distances_to_gt[pred_surface], distances_to_pred[gt_surface]))
    return float(np.percentile(surface_distances, 95))


def compute_binary_metrics(pred_mask, gt_mask, spacing_mm):
    """Return metrics with explicit semantics for empty regions.

    Dice and IoU are one when both masks are empty and zero when only one mask
    is empty. HD95 is ``None`` whenever a foreground surface is unavailable.
    Sensitivity is ``None`` for an empty ground-truth region because its
    denominator is zero; specificity remains defined for all masks.
    """
    pred = _as_mask(pred_mask, "pred_mask")
    gt = _as_mask(gt_mask, "gt_mask")
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shape mismatch: prediction {pred.shape}, ground truth {gt.shape}.")

    _validate_spacing(spacing_mm)
    intersection = int(np.count_nonzero(pred & gt))
    pred_voxels = int(np.count_nonzero(pred))
    gt_voxels = int(np.count_nonzero(gt))
    total_voxels = int(pred.size)
    union = pred_voxels + gt_voxels - intersection
    false_positive = pred_voxels - intersection
    false_negative = gt_voxels - intersection
    true_negative = total_voxels - intersection - false_positive - false_negative

    both_empty = pred_voxels == 0 and gt_voxels == 0
    if both_empty:
        dice = 1.0
        iou = 1.0
        sensitivity = None
        empty_region = "both_empty"
    elif gt_voxels == 0:
        dice = 0.0
        iou = 0.0
        sensitivity = None
        empty_region = "ground_truth_empty"
    elif pred_voxels == 0:
        dice = 0.0
        iou = 0.0
        sensitivity = 0.0
        empty_region = "prediction_empty"
    else:
        dice = float(2.0 * intersection / (pred_voxels + gt_voxels))
        iou = float(intersection / union)
        sensitivity = float(intersection / gt_voxels)
        empty_region = "none"

    specificity_denominator = true_negative + false_positive
    specificity = (
        float(true_negative / specificity_denominator)
        if specificity_denominator > 0
        else None
    )
    return {
        "dice": dice,
        "iou": iou,
        "hd95_mm": compute_hd95_mm(pred, gt, spacing_mm),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "pred_voxels": pred_voxels,
        "gt_voxels": gt_voxels,
        "intersection": intersection,
        "false_positive_voxels": false_positive,
        "false_negative_voxels": false_negative,
        "true_negative_voxels": true_negative,
        "empty_region": empty_region,
    }


def compute_hierarchy_violations(class_volumes: Mapping[str, np.ndarray]):
    """Count violations of ET subset TC subset WT on one native 3D grid."""
    masks = {class_name: _as_mask(class_volumes[class_name], class_name) for class_name in CLASS_NAMES}
    shapes = {mask.shape for mask in masks.values()}
    if len(shapes) != 1:
        raise ValueError(f"Class masks must have the same shape, got {sorted(shapes)}.")

    et_outside_tc = masks["ET"] & ~masks["TC"]
    tc_outside_wt = masks["TC"] & ~masks["WT"]
    any_violation = et_outside_tc | tc_outside_wt
    return {
        "et_outside_tc_voxels": int(np.count_nonzero(et_outside_tc)),
        "tc_outside_wt_voxels": int(np.count_nonzero(tc_outside_wt)),
        "any_violation_voxels": int(np.count_nonzero(any_violation)),
        "has_violation": bool(np.any(any_violation)),
    }


def evaluate_brats_case(class_volumes, gt_masks, spacing_mm):
    """Evaluate ET, TC and WT predictions against one BraTS case."""
    spacing = _validate_spacing(spacing_mm)
    per_class = {
        class_name: compute_binary_metrics(class_volumes[class_name], gt_masks[class_name], spacing)
        for class_name in CLASS_NAMES
    }
    return {
        "metric_schema_version": 1,
        "spacing_mm": list(spacing),
        "per_class": per_class,
        "mean_dice": float(np.mean([per_class[class_name]["dice"] for class_name in CLASS_NAMES])),
        "mean_iou": float(np.mean([per_class[class_name]["iou"] for class_name in CLASS_NAMES])),
        "hierarchy": compute_hierarchy_violations(class_volumes),
    }


def mean_defined(values: Sequence[Optional[float]]):
    defined = [float(value) for value in values if value is not None]
    return None if not defined else float(np.mean(defined))
