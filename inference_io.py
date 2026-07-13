"""Output and metadata helpers for BraTS volume inference."""

import json
from pathlib import Path

import numpy as np

from brats_constants import BRATS_CLASS_NAMES


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
    Path(output_path).write_text(json.dumps(_to_json_compatible(payload), indent=2), encoding="utf-8")


def save_mask_outputs(brats_case, output_dir, class_volumes, combined_label, prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"{prefix}_" if prefix else ""
    for class_name in BRATS_CLASS_NAMES:
        brats_case.save_nifti(
            class_volumes[class_name].astype(np.uint8),
            output_dir / f"{filename_prefix}{class_name}.nii.gz",
        )
    brats_case.save_nifti(combined_label.astype(np.uint8), output_dir / f"{filename_prefix}combined_label.nii.gz")


def save_case_meta(brats_case, output_dir, meta):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    brats_case.write_case_meta(output_dir, meta)


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
        "class_order": {str(index): name for index, name in enumerate(BRATS_CLASS_NAMES)},
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
        "modality_paths": {key: str(path.resolve()) for key, path in brats_case.modality_paths.items()},
        "segmentation_path": str(brats_case.segmentation_path.resolve()) if brats_case.segmentation_path else None,
        "postprocess": {
            **postprocess_config,
            "report_path": str(postprocess_report_path.resolve()) if postprocess_report_path else None,
        },
        "prompt_report_path": str(prompt_report_path.resolve()) if prompt_report_path else None,
        "yolo": yolo_config,
    }
