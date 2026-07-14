import logging

import numpy as np
from scipy import ndimage
from skimage.measure import label as connected_components

from sam_med2d_finetune.brats.constants import BRATS_CLASS_NAMES as CLASS_NAMES

LOGGER = logging.getLogger(__name__)


def _as_bool_mask(mask):
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={array.shape}.")
    return array.astype(bool, copy=False)


def _count_voxels(mask):
    return int(np.count_nonzero(mask))


def _make_structure(radius):
    if radius <= 0:
        return None
    base = ndimage.generate_binary_structure(rank=3, connectivity=1)
    return ndimage.iterate_structure(base, int(radius))


def apply_morphology_3d(mask, operation="closing", radius=1):
    mask_bool = _as_bool_mask(mask)
    if _count_voxels(mask_bool) == 0:
        LOGGER.info("Skipping morphology '%s' on an empty mask.", operation)
        return mask_bool.astype(np.uint8)
    if radius <= 0:
        return mask_bool.astype(np.uint8)

    structure = _make_structure(radius)
    if operation == "closing":
        processed = ndimage.binary_closing(mask_bool, structure=structure)
    elif operation == "opening":
        processed = ndimage.binary_opening(mask_bool, structure=structure)
    else:
        raise ValueError(f"Unsupported morphology operation: {operation}")
    return processed.astype(np.uint8)


def fill_holes_3d(mask):
    mask_bool = _as_bool_mask(mask)
    if _count_voxels(mask_bool) == 0:
        LOGGER.info("Skipping hole filling on an empty mask.")
        return mask_bool.astype(np.uint8)
    return ndimage.binary_fill_holes(mask_bool).astype(np.uint8)


def filter_connected_components(mask, keep_largest=False, top_k=1, min_voxels=0, connectivity=1):
    mask_bool = _as_bool_mask(mask)
    if _count_voxels(mask_bool) == 0:
        LOGGER.info("Skipping connected-component filtering on an empty mask.")
        return mask_bool.astype(np.uint8)

    labeled = connected_components(mask_bool, connectivity=connectivity)
    component_sizes = np.bincount(labeled.ravel())[1:]
    if component_sizes.size == 0:
        LOGGER.info("No foreground connected components found after labeling.")
        return np.zeros_like(mask_bool, dtype=np.uint8)

    candidates = []
    for component_id, size in enumerate(component_sizes, start=1):
        if int(size) >= int(min_voxels):
            candidates.append((component_id, int(size)))

    if not candidates:
        LOGGER.info("All connected components were removed by the min_voxels threshold.")
        return np.zeros_like(mask_bool, dtype=np.uint8)

    candidates.sort(key=lambda item: item[1], reverse=True)
    if keep_largest:
        kept_ids = [candidates[0][0]]
    elif top_k is None or int(top_k) <= 0:
        kept_ids = [component_id for component_id, _ in candidates]
    else:
        kept_ids = [component_id for component_id, _ in candidates[: int(top_k)]]

    filtered = np.isin(labeled, kept_ids)
    return filtered.astype(np.uint8)


def smooth_z_axis(mask, iterations=1):
    mask_bool = _as_bool_mask(mask)
    if _count_voxels(mask_bool) == 0:
        LOGGER.info("Skipping Z-axis smoothing on an empty mask.")
        return mask_bool.astype(np.uint8)

    result = mask_bool.copy()
    for _ in range(max(int(iterations), 0)):
        padded = np.pad(result, ((0, 0), (0, 0), (1, 1)), mode="constant", constant_values=False)
        prev_slice = padded[:, :, :-2]
        curr_slice = padded[:, :, 1:-1]
        next_slice = padded[:, :, 2:]

        supported_voxels = curr_slice & (prev_slice | next_slice)
        bridged_gaps = (~curr_slice) & prev_slice & next_slice
        result = supported_voxels | bridged_gaps

        if _count_voxels(result) == 0:
            LOGGER.info("Z-axis smoothing removed all foreground voxels.")
            break

    return result.astype(np.uint8)


def enforce_hierarchy(et_mask, tc_mask, wt_mask):
    et = _as_bool_mask(et_mask)
    tc = _as_bool_mask(tc_mask)
    wt = _as_bool_mask(wt_mask)

    tc = tc | et
    wt = wt | tc
    return et.astype(np.uint8), tc.astype(np.uint8), wt.astype(np.uint8)


def _record_step(report, step_name, mask):
    report[step_name] = _count_voxels(mask)


def _count_hierarchy_violations(et_mask, tc_mask, wt_mask):
    et = _as_bool_mask(et_mask)
    tc = _as_bool_mask(tc_mask)
    wt = _as_bool_mask(wt_mask)
    return {
        "et_outside_tc": int(np.count_nonzero(et & ~tc)),
        "tc_outside_wt": int(np.count_nonzero(tc & ~wt)),
    }


def postprocess_brats_masks(
    class_volumes,
    closing_radius=1,
    opening_radius=1,
    wt_keep_largest=True,
    keep_topk_tc=2,
    keep_topk_et=2,
    z_smooth_iterations=1,
):
    processed = {}
    report = {
        "config": {
            "closing_radius": int(closing_radius),
            "opening_radius": int(opening_radius),
            "wt_keep_largest": bool(wt_keep_largest),
            "keep_topk_tc": int(keep_topk_tc),
            "keep_topk_et": int(keep_topk_et),
            "z_smooth_iterations": int(z_smooth_iterations),
        },
        "classes": {},
    }

    for class_name in CLASS_NAMES:
        mask = _as_bool_mask(class_volumes[class_name]).astype(np.uint8)
        class_report = {}
        _record_step(class_report, "raw", mask)

        closed = apply_morphology_3d(mask, operation="closing", radius=closing_radius)
        _record_step(class_report, "after_closing", closed)

        opened = apply_morphology_3d(closed, operation="opening", radius=opening_radius)
        _record_step(class_report, "after_opening", opened)

        filled = fill_holes_3d(opened)
        _record_step(class_report, "after_fill_holes", filled)

        smoothed = smooth_z_axis(filled, iterations=z_smooth_iterations)
        _record_step(class_report, "after_z_smoothing", smoothed)

        if class_name == "WT":
            filtered = filter_connected_components(
                smoothed,
                keep_largest=wt_keep_largest,
                top_k=None,
            )
        elif class_name == "TC":
            filtered = filter_connected_components(smoothed, top_k=keep_topk_tc)
        else:
            filtered = filter_connected_components(smoothed, top_k=keep_topk_et)

        _record_step(class_report, "after_component_filter", filtered)
        processed[class_name] = filtered.astype(np.uint8)
        report["classes"][class_name] = class_report

    report["hierarchy"] = {
        "before": _count_hierarchy_violations(processed["ET"], processed["TC"], processed["WT"])
    }
    et_mask, tc_mask, wt_mask = enforce_hierarchy(processed["ET"], processed["TC"], processed["WT"])
    processed["ET"] = et_mask
    processed["TC"] = tc_mask
    processed["WT"] = wt_mask
    report["hierarchy"]["after"] = _count_hierarchy_violations(
        processed["ET"], processed["TC"], processed["WT"]
    )

    for class_name in CLASS_NAMES:
        _record_step(report["classes"][class_name], "after_hierarchy", processed[class_name])

    return processed, report
