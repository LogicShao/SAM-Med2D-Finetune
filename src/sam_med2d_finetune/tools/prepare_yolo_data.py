import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sam_med2d_finetune.brats.case import BRATS_MODALITIES, BraTSCase, YOLO_MODALITIES
from sam_med2d_finetune.brats.cache import CACHE_NORMALIZATION


CLASS_ID = 0
CLASS_NAME = "Tumor"
DEFAULT_SPLITS = ("train", "val")
CASE_PREFIX = "BraTS2021_"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare one-class YOLO WT-box data from a frozen BraTS split."
    )
    parser.add_argument("--split_root", required=True, help="Root containing train/val BraTS case directories.")
    parser.add_argument("--out_dir", required=True, help="YOLO dataset output directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
        help="Splits to export. Test export is intentionally unsupported.",
    )
    parser.add_argument("--seed", type=int, default=11171, help="Seed for deterministic negative sampling.")
    parser.add_argument(
        "--negative_to_positive_ratio",
        type=float,
        default=1.0 / 3.0,
        help="Training negative slice ratio relative to positive slices.",
    )
    parser.add_argument(
        "--box_padding_ratio",
        type=float,
        default=0.10,
        help="Padding applied to every WT box side as a fraction of box width/height.",
    )
    parser.add_argument("--max_cases_per_split", type=int, default=None, help="Optional smoke-test case cap per split.")
    parser.add_argument("--clean", action="store_true", help="Recreate out_dir before writing.")
    args = parser.parse_args()

    if args.negative_to_positive_ratio < 0.0:
        parser.error("--negative_to_positive_ratio must be zero or greater.")
    if args.box_padding_ratio < 0.0:
        parser.error("--box_padding_ratio must be zero or greater.")
    if args.max_cases_per_split is not None and args.max_cases_per_split <= 0:
        parser.error("--max_cases_per_split must be greater than zero when provided.")
    return args


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(split_root):
    manifest_path = Path(split_root) / "split_manifest.json"
    if not manifest_path.is_file():
        return None, None
    return json.loads(manifest_path.read_text(encoding="utf-8")), sha256_file(manifest_path)


def ensure_output_root(out_dir, clean):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not clean:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --clean to recreate it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def list_case_dirs(split_root, split_name, max_cases=None):
    split_dir = Path(split_root) / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    case_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir() and path.name.startswith(CASE_PREFIX))
    if max_cases is not None:
        case_dirs = case_dirs[: int(max_cases)]
    if not case_dirs:
        raise ValueError(f"No BraTS cases found for split {split_name}: {split_dir}")
    return case_dirs


def _find_single_file(case_dir, suffix):
    matches = sorted(Path(case_dir).glob(f"*_{suffix}.nii.gz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one '*_{suffix}.nii.gz' file in {case_dir}, found {len(matches)}."
        )
    return matches[0]


def validate_case_files(case_dir):
    for modality in BRATS_MODALITIES:
        _find_single_file(case_dir, modality)
    _find_single_file(case_dir, "seg")


def collect_case_dirs_for_splits(split_root, splits, max_cases_per_split=None):
    split_cases = {
        split_name: list_case_dirs(split_root, split_name, max_cases_per_split)
        for split_name in splits
    }
    seen = {}
    for split_name, case_dirs in split_cases.items():
        for case_dir in case_dirs:
            previous_split = seen.get(case_dir.name)
            if previous_split is not None:
                raise ValueError(
                    f"Duplicate case ID across splits: {case_dir.name} in {previous_split} and {split_name}."
                )
            seen[case_dir.name] = split_name
            validate_case_files(case_dir)
    return split_cases


def validate_against_source_manifest(split_cases, source_manifest, max_cases_per_split=None):
    if not source_manifest:
        return
    manifest_splits = source_manifest.get("splits", {})
    for split_name, case_dirs in split_cases.items():
        if split_name not in manifest_splits:
            raise ValueError(f"Source manifest does not define split: {split_name}")
        manifest_case_ids = set(manifest_splits[split_name].get("case_ids", []))
        selected_case_ids = {case_dir.name for case_dir in case_dirs}
        missing_from_manifest = sorted(selected_case_ids - manifest_case_ids)
        if missing_from_manifest:
            raise ValueError(
                f"{split_name} contains cases absent from source manifest: {missing_from_manifest[:5]}"
            )
        if max_cases_per_split is None and selected_case_ids != manifest_case_ids:
            missing_on_disk = sorted(manifest_case_ids - selected_case_ids)
            extra_on_disk = sorted(selected_case_ids - manifest_case_ids)
            raise ValueError(
                f"{split_name} case IDs do not match source manifest. "
                f"missing_on_disk={missing_on_disk[:5]}, extra_on_disk={extra_on_disk[:5]}"
            )

    selected_by_split = {
        split_name: {case_dir.name for case_dir in case_dirs}
        for split_name, case_dirs in split_cases.items()
    }
    if "train" in selected_by_split and "val" in selected_by_split:
        overlap = sorted(selected_by_split["train"] & selected_by_split["val"])
        if overlap:
            raise ValueError(f"Train/val overlap is not allowed: {overlap[:5]}")


def wt_box_xyxy(mask_2d, padding_ratio):
    y_indices, x_indices = np.where(mask_2d > 0)
    if y_indices.size == 0:
        return None

    height, width = mask_2d.shape
    x1 = float(x_indices.min())
    y1 = float(y_indices.min())
    x2 = float(x_indices.max())
    y2 = float(y_indices.max())
    box_width = max(x2 - x1 + 1.0, 1.0)
    box_height = max(y2 - y1 + 1.0, 1.0)
    pad_x = box_width * float(padding_ratio)
    pad_y = box_height * float(padding_ratio)
    return [
        float(np.clip(x1 - pad_x, 0.0, width - 1.0)),
        float(np.clip(y1 - pad_y, 0.0, height - 1.0)),
        float(np.clip(x2 + pad_x, 0.0, width - 1.0)),
        float(np.clip(y2 + pad_y, 0.0, height - 1.0)),
    ]


def xyxy_to_yolo_line(box, image_shape):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    box_width = max(x2 - x1 + 1.0, 1.0)
    box_height = max(y2 - y1 + 1.0, 1.0)
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    values = [
        CLASS_ID,
        x_center / float(width),
        y_center / float(height),
        box_width / float(width),
        box_height / float(height),
    ]
    return "{} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(*values)


def select_train_slices(case_id, positive_slices, negative_slices, ratio, seed):
    positive_slices = list(positive_slices)
    negative_slices = list(negative_slices)
    negative_count = min(len(negative_slices), int(round(len(positive_slices) * float(ratio))))
    seed_material = f"{int(seed)}:{case_id}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], byteorder="big", signed=False)
    rng = random.Random(stable_seed)
    selected_negative = sorted(rng.sample(negative_slices, negative_count)) if negative_count > 0 else []
    return sorted(positive_slices + selected_negative), selected_negative


def write_sample(brats_case, split_name, slice_index, box, out_dir):
    image_dir = out_dir / "images" / split_name
    label_dir = out_dir / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{brats_case.case_id}_z{int(slice_index):03d}"
    image = brats_case.get_pseudo_rgb_slice(slice_index)
    image_path = image_dir / f"{stem}.png"
    label_path = label_dir / f"{stem}.txt"

    Image.fromarray(image).save(image_path)
    if box is None:
        label_path.write_text("", encoding="utf-8")
    else:
        label_path.write_text(xyxy_to_yolo_line(box, image.shape), encoding="utf-8")
    return image_path, label_path


def export_split(case_dirs, out_dir, split_name, args):
    split_records = []
    positive_slices_total = 0
    negative_slices_total = 0
    exported_positive = 0
    exported_negative = 0

    for case_dir in case_dirs:
        brats_case = BraTSCase.from_dir(case_dir)
        if not brats_case.has_segmentation():
            raise ValueError(f"Segmentation is required for YOLO labels: {case_dir}")

        wt_volume = brats_case.class_gt_volumes["WT"]
        positive_slices = [index for index in range(brats_case.shape[2]) if np.any(wt_volume[:, :, index] > 0)]
        negative_slices = [index for index in range(brats_case.shape[2]) if not np.any(wt_volume[:, :, index] > 0)]
        positive_slices_total += len(positive_slices)
        negative_slices_total += len(negative_slices)

        if split_name == "train":
            selected_slices, selected_negative = select_train_slices(
                brats_case.case_id,
                positive_slices,
                negative_slices,
                args.negative_to_positive_ratio,
                args.seed,
            )
        else:
            selected_slices = list(range(brats_case.shape[2]))
            selected_negative = negative_slices

        if len(selected_slices) != len(set(selected_slices)):
            raise ValueError(f"Duplicate slice IDs selected for {brats_case.case_id} in {split_name}.")

        for slice_index in selected_slices:
            mask = wt_volume[:, :, int(slice_index)]
            box = wt_box_xyxy(mask, args.box_padding_ratio)
            write_sample(brats_case, split_name, slice_index, box, out_dir)
            if box is None:
                exported_negative += 1
            else:
                exported_positive += 1

        split_records.append(
            {
                "case_id": brats_case.case_id,
                "shape": list(brats_case.shape),
                "positive_slices": positive_slices,
                "negative_slices": negative_slices,
                "selected_slices": selected_slices,
                "selected_negative_slices": selected_negative,
            }
        )

    return {
        "case_count": len(case_dirs),
        "positive_slices_source": positive_slices_total,
        "negative_slices_source": negative_slices_total,
        "exported_positive_slices": exported_positive,
        "exported_negative_slices": exported_negative,
        "exported_slices": exported_positive + exported_negative,
        "cases": split_records,
    }


def write_data_yaml(out_dir):
    data = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {CLASS_ID: CLASS_NAME},
    }
    data_yaml = Path(out_dir) / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_yaml


def main():
    args = parse_args()
    split_root = Path(args.split_root).resolve()
    source_manifest, source_manifest_sha256 = load_split_manifest(split_root)
    split_cases = collect_case_dirs_for_splits(split_root, args.splits, args.max_cases_per_split)
    validate_against_source_manifest(split_cases, source_manifest, args.max_cases_per_split)
    out_dir = ensure_output_root(args.out_dir, args.clean)

    exported = {}
    for split_name in args.splits:
        exported[split_name] = export_split(split_cases[split_name], out_dir, split_name, args)

    data_yaml = write_data_yaml(out_dir)
    manifest = {
        "schema_version": 1,
        "dataset": "brats_yolo_wt_box",
        "source_split_root": str(split_root),
        "source_split_manifest_sha256": source_manifest_sha256,
        "source_split_manifest": source_manifest,
        "out_dir": str(out_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "splits": list(args.splits),
        "seed": int(args.seed),
        "negative_to_positive_ratio": float(args.negative_to_positive_ratio),
        "box_padding_ratio": float(args.box_padding_ratio),
        "modalities": list(YOLO_MODALITIES),
        "normalization": CACHE_NORMALIZATION,
        "box_source": "WT: seg > 0, padded and clipped in native slice coordinates",
        "class_map": {str(CLASS_ID): CLASS_NAME},
        "exports": exported,
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary = {
        "out_dir": str(out_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "dataset_manifest": str(manifest_path.resolve()),
        "splits": {
            split_name: {
                key: value
                for key, value in split_summary.items()
                if key != "cases"
            }
            for split_name, split_summary in exported.items()
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
