import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm


PSEUDO_RGB_MODALITIES = ("t1ce", "t2", "flair")
DEFAULT_SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a YOLO 2D detection dataset from BraTS NIfTI volumes.")
    parser.add_argument(
        "--raw_dir",
        default="data_brats_raw",
        help="BraTS raw dataset root containing train/, val/, and optional test/ case directories.",
    )
    parser.add_argument(
        "--out_dir",
        default="datasets/brats_yolo",
        help="Output directory for the YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--bg_ratio",
        type=float,
        default=0.10,
        help="Sampling ratio for healthy background slices. Range: 0.0 to 1.0.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed used for deterministic background sampling.",
    )
    return parser.parse_args()


def _find_single_file(case_dir, suffix):
    matches = sorted(case_dir.glob(f"*_{suffix}.nii.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one '*_{suffix}.nii.gz' file in {case_dir}, found {len(matches)}.")
    return matches[0]


def _normalize_volume_to_uint8(volume):
    volume = volume.astype(np.float32, copy=False)
    nonzero_mask = volume != 0
    normalized = np.zeros_like(volume, dtype=np.uint8)
    if not np.any(nonzero_mask):
        return normalized

    valid = volume[nonzero_mask]
    min_value = float(valid.min())
    max_value = float(valid.max())
    scale = max(max_value - min_value, 1e-8)
    normalized_float = np.zeros_like(volume, dtype=np.float32)
    normalized_float[nonzero_mask] = (volume[nonzero_mask] - min_value) / scale
    normalized = np.clip(normalized_float * 255.0, 0, 255).astype(np.uint8)
    return normalized


def _load_case_volumes(case_dir):
    case_dir = Path(case_dir)
    case_id = case_dir.name

    modality_volumes = {}
    reference_shape = None
    for modality in PSEUDO_RGB_MODALITIES:
        path = _find_single_file(case_dir, modality)
        volume = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
        if reference_shape is None:
            reference_shape = volume.shape
        elif volume.shape != reference_shape:
            raise ValueError(f"Shape mismatch in {case_id}: {modality} has shape {volume.shape}, expected {reference_shape}.")
        modality_volumes[modality] = _normalize_volume_to_uint8(volume)

    seg_path = _find_single_file(case_dir, "seg")
    seg_volume = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.int16)
    if seg_volume.shape != reference_shape:
        raise ValueError(f"Shape mismatch in {case_id}: seg has shape {seg_volume.shape}, expected {reference_shape}.")

    return modality_volumes, seg_volume


def _slice_to_png(modality_volumes, slice_index):
    channels = [modality_volumes[modality][:, :, slice_index] for modality in PSEUDO_RGB_MODALITIES]
    return np.stack(channels, axis=-1)


def _mask_to_yolo_bbox(mask_2d):
    y_indices, x_indices = np.where(mask_2d > 0)
    if y_indices.size == 0:
        return None

    height, width = mask_2d.shape
    x_min = int(x_indices.min())
    x_max = int(x_indices.max())
    y_min = int(y_indices.min())
    y_max = int(y_indices.max())

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1
    x_center = x_min + bbox_width / 2.0
    y_center = y_min + bbox_height / 2.0

    return (
        0,
        x_center / width,
        y_center / height,
        bbox_width / width,
        bbox_height / height,
    )


def _case_seed(case_id, global_seed):
    digest = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(global_seed)) % (2 ** 32)


def _process_case(case_dir, image_dir, label_dir, bg_ratio, seed):
    case_dir = Path(case_dir)
    case_id = case_dir.name
    modality_volumes, seg_volume = _load_case_volumes(case_dir)
    rng = np.random.default_rng(_case_seed(case_id, seed))

    saved_positive = 0
    saved_background = 0
    total_slices = int(seg_volume.shape[2])

    for slice_index in range(total_slices):
        seg_slice = seg_volume[:, :, slice_index]
        has_tumor = bool(np.any(seg_slice > 0))
        should_keep = has_tumor or (rng.random() < float(bg_ratio))
        if not should_keep:
            continue

        image = _slice_to_png(modality_volumes, slice_index)
        stem = f"{case_id}_z{slice_index:03d}"
        image_path = Path(image_dir) / f"{stem}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if has_tumor:
            bbox = _mask_to_yolo_bbox(seg_slice > 0)
            if bbox is None:
                continue
            label_path = Path(label_dir) / f"{stem}.txt"
            label_path.write_text(
                f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n",
                encoding="utf-8",
            )
            saved_positive += 1
        else:
            saved_background += 1

    return {
        "case_id": case_id,
        "total_slices": total_slices,
        "saved_positive": saved_positive,
        "saved_background": saved_background,
    }

def _prepare_output_dirs(out_dir, splits):
    out_dir = Path(out_dir)
    paths = {}
    for split in splits:
        image_dir = out_dir / "images" / split
        label_dir = out_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        paths[split] = {"images": image_dir, "labels": label_dir}
    return paths


def _discover_splits(raw_dir):
    raw_dir = Path(raw_dir)
    splits = [split for split in DEFAULT_SPLITS if (raw_dir / split).is_dir()]
    if not splits:
        raise FileNotFoundError(
            f"No split directories found under {raw_dir}. Expected train/, val/, and optional test/."
        )
    if "train" not in splits or "val" not in splits:
        raise FileNotFoundError(
            f"Both train/ and val/ must exist under {raw_dir}. Found splits: {splits}."
        )
    return splits


def _write_data_yaml(out_dir, splits):
    out_dir = Path(out_dir)
    lines = [
        "train: images/train",
        "val: images/val",
    ]
    if "test" in splits:
        lines.append("test: images/test")
    lines.extend(
        [
            "",
            "names:",
            "  0: Tumor",
            "",
        ]
    )
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def _process_split(case_dirs, image_dir, label_dir, bg_ratio, seed, workers, split_name):
    results = []
    workers = max(1, int(workers))

    if workers == 1:
        iterator = case_dirs
        for case_dir in tqdm(iterator, total=len(case_dirs), desc=f"Prepare {split_name}"):
            results.append(
                _process_case(
                    case_dir=case_dir,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    bg_ratio=bg_ratio,
                    seed=seed,
                )
            )
        return results

    try:
        futures = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for case_dir in case_dirs:
                futures.append(
                    executor.submit(
                        _process_case,
                        case_dir=case_dir,
                        image_dir=image_dir,
                        label_dir=label_dir,
                        bg_ratio=bg_ratio,
                        seed=seed,
                    )
                )

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Prepare {split_name}"):
                results.append(future.result())
        return results
    except PermissionError:
        print(f"Parallel processing is unavailable for split '{split_name}'. Falling back to workers=1.")
        return _process_split(
            case_dirs=case_dirs,
            image_dir=image_dir,
            label_dir=label_dir,
            bg_ratio=bg_ratio,
            seed=seed,
            workers=1,
            split_name=split_name,
        )


def main():
    args = parse_args()
    if not 0.0 <= float(args.bg_ratio) <= 1.0:
        raise ValueError("--bg_ratio must be between 0.0 and 1.0.")

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    splits = _discover_splits(raw_dir)
    output_dirs = _prepare_output_dirs(out_dir, splits)

    all_results = {}
    for split in splits:
        split_dir = raw_dir / split
        case_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        results = _process_split(
            case_dirs=case_dirs,
            image_dir=output_dirs[split]["images"],
            label_dir=output_dirs[split]["labels"],
            bg_ratio=args.bg_ratio,
            seed=args.seed,
            workers=args.workers,
            split_name=split,
        )

        results.sort(key=lambda item: item["case_id"])
        all_results[split] = {
            "cases": results,
            "num_cases": len(results),
            "saved_positive": int(sum(item["saved_positive"] for item in results)),
            "saved_background": int(sum(item["saved_background"] for item in results)),
        }

    yaml_path = _write_data_yaml(out_dir, splits)
    summary = {
        "raw_dir": str(raw_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "bg_ratio": float(args.bg_ratio),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "processed_splits": splits,
        "splits": all_results,
        "data_yaml": str(yaml_path.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
