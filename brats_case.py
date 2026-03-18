import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np


BRATS_MODALITIES = ("t1", "t1ce", "t2", "flair")


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


def _normalize_volume(volume):
    volume = volume.astype(np.float32, copy=False)
    mask = volume != 0
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32), {"min": 0.0, "max": 0.0, "nonzero_voxels": 0}

    valid_values = volume[mask]
    min_value = float(valid_values.min())
    max_value = float(valid_values.max())
    scale = max(max_value - min_value, 1e-8)

    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - min_value) / scale
    return normalized, {
        "min": min_value,
        "max": max_value,
        "nonzero_voxels": int(mask.sum()),
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
        )

    def get_slice_tensor(self, slice_index, image_size):
        channels = []
        for modality in BRATS_MODALITIES:
            slice_2d = self.normalized_volumes[modality][:, :, slice_index]
            resized = cv2.resize(slice_2d, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            channels.append(resized)
        return np.stack(channels, axis=0).astype(np.float32)

    def save_nifti(self, volume, output_path):
        header = self.header.copy()
        header.set_data_dtype(volume.dtype)
        image = nib.Nifti1Image(volume, self.affine, header=header)
        nib.save(image, str(output_path))

    def write_case_meta(self, output_dir, meta):
        meta_path = Path(output_dir) / "case_meta.json"
        json_compatible_meta = _to_json_compatible(meta)
        meta_path.write_text(json.dumps(json_compatible_meta, indent=2), encoding="utf-8")
