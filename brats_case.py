import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import nibabel as nib
import numpy as np

from brats_cache import normalize_nonzero_volume
from brats_constants import BRATS_CLASS_NAMES as CLASS_NAMES

BRATS_MODALITIES = ("t1", "t1ce", "t2", "flair")
YOLO_MODALITIES = ("t1ce", "t2", "flair")


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


def _find_modality_file(case_dir, modality):
    matches = sorted(case_dir.glob(f"*_{modality}.nii.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*_{modality}.nii.gz' file in {case_dir}, found {len(matches)}."
        )
    return matches[0]


def _find_optional_segmentation_file(case_dir):
    matches = sorted(case_dir.glob("*_seg.nii.gz"))
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Expected at most one '*_seg.nii.gz' file in {case_dir}, found {len(matches)}."
        )
    return matches[0] if matches else None


def _normalize_volume(volume):
    return normalize_nonzero_volume(volume)


def _build_gt_class_volumes(segmentation_volume):
    return {
        "ET": (segmentation_volume == 4).astype(np.uint8),
        "TC": np.isin(segmentation_volume, [1, 4]).astype(np.uint8),
        "WT": np.isin(segmentation_volume, [1, 2, 4]).astype(np.uint8),
    }


@dataclass
class BraTSCase:
    case_dir: Path
    case_id: str
    modality_paths: dict
    modality_volumes: dict
    normalized_volumes: dict
    normalization_stats: dict
    affine: np.ndarray
    header: nib.Nifti1Header
    shape: tuple
    segmentation_path: Optional[Path] = None
    segmentation_volume: Optional[np.ndarray] = None
    class_gt_volumes: Optional[dict] = None

    @classmethod
    def from_dir(cls, case_dir):
        case_dir = Path(case_dir)
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        modality_paths = {modality: _find_modality_file(case_dir, modality) for modality in BRATS_MODALITIES}
        images = {modality: nib.load(str(path)) for modality, path in modality_paths.items()}

        reference_modality = BRATS_MODALITIES[0]
        reference_image = images[reference_modality]
        reference_shape = reference_image.shape
        reference_affine = reference_image.affine

        modality_volumes = {}
        normalized_volumes = {}
        normalization_stats = {}
        segmentation_path = _find_optional_segmentation_file(case_dir)
        segmentation_volume = None
        class_gt_volumes = None

        for modality, image in images.items():
            if image.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch for {modality}: expected {reference_shape}, got {image.shape}."
                )
            if not np.allclose(image.affine, reference_affine):
                raise ValueError(f"Affine mismatch for modality {modality} in {case_dir}.")

            volume = np.asarray(image.get_fdata(dtype=np.float32))
            modality_volumes[modality] = volume
            normalized_volumes[modality], normalization_stats[modality] = _normalize_volume(volume)

        if segmentation_path is not None:
            segmentation_image = nib.load(str(segmentation_path))
            if segmentation_image.shape != reference_shape:
                raise ValueError(
                    f"Shape mismatch for segmentation: expected {reference_shape}, got {segmentation_image.shape}."
                )
            if not np.allclose(segmentation_image.affine, reference_affine):
                raise ValueError(f"Affine mismatch for segmentation in {case_dir}.")
            segmentation_volume = np.asarray(segmentation_image.dataobj, dtype=np.int16)
            class_gt_volumes = _build_gt_class_volumes(segmentation_volume)

        return cls(
            case_dir=case_dir,
            case_id=case_dir.name,
            modality_paths=modality_paths,
            modality_volumes=modality_volumes,
            normalized_volumes=normalized_volumes,
            normalization_stats=normalization_stats,
            affine=reference_affine.copy(),
            header=reference_image.header.copy(),
            shape=reference_shape,
            segmentation_path=segmentation_path,
            segmentation_volume=segmentation_volume,
            class_gt_volumes=class_gt_volumes,
        )

    def get_slice_tensor(self, slice_index, image_size):
        channels = []
        for modality in BRATS_MODALITIES:
            slice_2d = self.normalized_volumes[modality][:, :, slice_index]
            resized = cv2.resize(slice_2d, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            channels.append(resized)
        return np.stack(channels, axis=0).astype(np.float32)

    def get_pseudo_rgb_slice(self, slice_index):
        channels = []
        for modality in YOLO_MODALITIES:
            slice_2d = self.normalized_volumes[modality][:, :, slice_index]
            channels.append(np.clip(slice_2d * 255.0, 0, 255).astype(np.uint8))
        return np.stack(channels, axis=-1)

    def save_nifti(self, volume, output_path):
        header = self.header.copy()
        header.set_data_dtype(volume.dtype)
        image = nib.Nifti1Image(volume, self.affine, header=header)
        nib.save(image, str(output_path))

    def write_case_meta(self, output_dir, meta):
        meta_path = Path(output_dir) / "case_meta.json"
        json_compatible_meta = _to_json_compatible(meta)
        meta_path.write_text(json.dumps(json_compatible_meta, indent=2), encoding="utf-8")

    def has_segmentation(self):
        return self.segmentation_volume is not None

    def slice_has_any_gt(self, slice_index):
        if not self.class_gt_volumes:
            return False
        return any(np.any(self.class_gt_volumes[class_name][:, :, slice_index]) for class_name in CLASS_NAMES)

    def get_gt_mask_slice(self, class_name, slice_index, image_size=None):
        if not self.class_gt_volumes:
            raise ValueError("Ground-truth segmentation is not available for this case.")
        if class_name not in self.class_gt_volumes:
            raise KeyError(f"Unsupported class_name: {class_name}")

        mask_slice = self.class_gt_volumes[class_name][:, :, slice_index].astype(np.uint8)
        if image_size is None or mask_slice.shape == (image_size, image_size):
            return mask_slice
        return cv2.resize(mask_slice, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    def get_gt_box(self, class_name, slice_index, image_size):
        mask_slice = self.get_gt_mask_slice(class_name, slice_index, image_size=image_size)
        y_indices, x_indices = np.where(mask_slice > 0)
        if y_indices.size == 0:
            return None
        return [
            float(x_indices.min()),
            float(y_indices.min()),
            float(x_indices.max()),
            float(y_indices.max()),
        ]
