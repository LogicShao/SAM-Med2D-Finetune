"""Shared BraTS cache contract and per-volume normalization."""

import json
from pathlib import Path

import numpy as np


CACHE_SCHEMA_VERSION = 1
CACHE_NORMALIZATION = "per_volume_nonzero_minmax_v1"
CACHE_IMAGES_FILENAME = "images.npy"
CACHE_SEGMENTATION_FILENAME = "seg.npy"
CACHE_IMAGES_DTYPE = np.dtype(np.float16)
CACHE_SEGMENTATION_DTYPE = np.dtype(np.uint8)
CACHE_MODALITY_COUNT = 4


def normalize_nonzero_volume(volume):
    volume = np.asarray(volume, dtype=np.float32)
    nonzero_mask = volume != 0
    normalized = np.zeros_like(volume, dtype=np.float32)
    if not np.any(nonzero_mask):
        return normalized, {"min": 0.0, "max": 0.0, "nonzero_voxels": 0}

    values = volume[nonzero_mask]
    min_value = float(values.min())
    max_value = float(values.max())
    normalized[nonzero_mask] = (values - min_value) / max(max_value - min_value, 1e-8)
    return normalized, {
        "min": min_value,
        "max": max_value,
        "nonzero_voxels": int(nonzero_mask.sum()),
    }


def validate_cache_case(cache_root, case_id):
    cache_dir = Path(cache_root) / case_id
    metadata_path = cache_dir / "metadata.json"
    images_path = cache_dir / CACHE_IMAGES_FILENAME
    segmentation_path = cache_dir / CACHE_SEGMENTATION_FILENAME
    missing_paths = [path for path in (metadata_path, images_path, segmentation_path) if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing cache files for {}: {}".format(case_id, missing_paths))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_values = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "case_id": case_id,
        "images_file": CACHE_IMAGES_FILENAME,
        "segmentation_file": CACHE_SEGMENTATION_FILENAME,
        "images_dtype": str(CACHE_IMAGES_DTYPE),
        "segmentation_dtype": str(CACHE_SEGMENTATION_DTYPE),
        "normalization": CACHE_NORMALIZATION,
    }
    for key, expected_value in expected_values.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                "Incompatible cache metadata for {}: {}={!r}, expected {!r}".format(
                    case_id, key, metadata.get(key), expected_value
                )
            )

    images = np.load(str(images_path), mmap_mode="r")
    segmentation = np.load(str(segmentation_path), mmap_mode="r")
    if images.dtype != CACHE_IMAGES_DTYPE or segmentation.dtype != CACHE_SEGMENTATION_DTYPE:
        raise ValueError("Incompatible cache dtypes for {}.".format(case_id))
    if images.ndim != 4 or images.shape[0] != CACHE_MODALITY_COUNT or segmentation.ndim != 3:
        raise ValueError("Incompatible cache dimensions for {}.".format(case_id))
    if tuple(images.shape[1:]) != tuple(segmentation.shape):
        raise ValueError("Image/segmentation shape mismatch in cache for {}.".format(case_id))
    if metadata.get("image_shape") != list(images.shape) or metadata.get("segmentation_shape") != list(segmentation.shape):
        raise ValueError("Metadata shape mismatch in cache for {}.".format(case_id))
    return {"metadata": metadata, "images": images, "segmentation": segmentation}
