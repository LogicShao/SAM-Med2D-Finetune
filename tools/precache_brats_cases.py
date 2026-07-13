import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brats_cache import (
    CACHE_IMAGES_DTYPE,
    CACHE_IMAGES_FILENAME,
    CACHE_NORMALIZATION,
    CACHE_SCHEMA_VERSION,
    CACHE_SEGMENTATION_DTYPE,
    CACHE_SEGMENTATION_FILENAME,
    normalize_nonzero_volume,
)

MODALITIES = ("t1", "t1ce", "t2", "flair")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a memory-mappable per-case BraTS training cache.")
    parser.add_argument("--cases_root", required=True, help="Directory containing BraTS case folders.")
    parser.add_argument("--cache_root", required=True, help="Directory where cache files will be written.")
    parser.add_argument("--case_ids", nargs="+", default=None, help="Optional explicit case IDs to cache.")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional maximum number of sorted cases.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files for selected cases.")
    return parser.parse_args()


def find_case_dirs(cases_root, case_ids=None, max_cases=None):
    root = Path(cases_root)
    if not root.is_dir():
        raise FileNotFoundError("Cases root not found: {}".format(root))
    if case_ids:
        case_dirs = [root / case_id for case_id in case_ids]
    else:
        case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    missing = [str(path) for path in case_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Case directories not found: {}".format(missing))
    return case_dirs if max_cases is None else case_dirs[: max(int(max_cases), 0)]


def build_case_cache(case_dir, cache_root, overwrite=False):
    case_dir = Path(case_dir)
    case_id = case_dir.name
    output_dir = Path(cache_root) / case_id
    images_path = output_dir / CACHE_IMAGES_FILENAME
    segmentation_path = output_dir / CACHE_SEGMENTATION_FILENAME
    metadata_path = output_dir / "metadata.json"

    if images_path.is_file() and segmentation_path.is_file() and metadata_path.is_file() and not overwrite:
        return "skipped"

    images = []
    normalization = {}
    reference_shape = None
    for modality in MODALITIES:
        source_path = case_dir / "{}_{}.nii.gz".format(case_id, modality)
        if not source_path.is_file():
            raise FileNotFoundError("Missing modality: {}".format(source_path))
        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(source_path)))
        normalized, stats = normalize_nonzero_volume(volume)
        if reference_shape is None:
            reference_shape = normalized.shape
        elif normalized.shape != reference_shape:
            raise ValueError("Shape mismatch in {}: {} != {}".format(source_path, normalized.shape, reference_shape))
        images.append(normalized.astype(CACHE_IMAGES_DTYPE))
        normalization[modality] = stats

    source_segmentation = case_dir / "{}_seg.nii.gz".format(case_id)
    if not source_segmentation.is_file():
        raise FileNotFoundError("Missing segmentation: {}".format(source_segmentation))
    segmentation = sitk.GetArrayFromImage(sitk.ReadImage(str(source_segmentation))).astype(CACHE_SEGMENTATION_DTYPE)
    if segmentation.shape != reference_shape:
        raise ValueError("Segmentation shape mismatch in {}".format(source_segmentation))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(images_path), np.stack(images, axis=0))
    np.save(str(segmentation_path), segmentation)
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "case_id": case_id,
                "source_case_dir": str(case_dir.resolve()),
                "images_file": images_path.name,
                "segmentation_file": segmentation_path.name,
                "image_shape": [int(value) for value in (len(images),) + reference_shape],
                "segmentation_shape": [int(value) for value in segmentation.shape],
                "images_dtype": str(CACHE_IMAGES_DTYPE),
                "segmentation_dtype": str(CACHE_SEGMENTATION_DTYPE),
                "normalization": CACHE_NORMALIZATION,
                "normalization_stats": normalization,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return "written"


def main():
    args = parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max_cases must be greater than zero when provided.")
    case_dirs = find_case_dirs(args.cases_root, args.case_ids, args.max_cases)
    counts = {"written": 0, "skipped": 0}
    for index, case_dir in enumerate(case_dirs, start=1):
        status = build_case_cache(case_dir, args.cache_root, overwrite=args.overwrite)
        counts[status] += 1
        print("[{}/{}] {}: {}".format(index, len(case_dirs), case_dir.name, status))
    print(json.dumps({"cache_root": str(Path(args.cache_root).resolve()), "cases": len(case_dirs), **counts}, indent=2))


if __name__ == "__main__":
    main()
