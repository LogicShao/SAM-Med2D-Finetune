import argparse
import json
import math
import random
import shutil
from pathlib import Path


CASE_PREFIX = "BraTS2021_"
DEFAULT_SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split BraTS raw cases into train/val/test and optionally create a small dev subset."
    )
    parser.add_argument(
        "--source_dir",
        default="data_brats_raw_all",
        help="Directory containing all BraTS case folders or existing split subdirectories.",
    )
    parser.add_argument(
        "--out_dir",
        default="data_brats_raw",
        help="Output directory for the full train/val/test split.",
    )
    parser.add_argument(
        "--dev_out_dir",
        default="data_brats_dev",
        help="Output directory for the sampled dev subset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.70,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic shuffling.",
    )
    parser.add_argument(
        "--dev_size",
        type=int,
        default=200,
        help="Number of train cases to sample into the dev dataset. Use 0 to disable dev export.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing out_dir/dev_out_dir before writing new split results.",
    )
    return parser.parse_args()


def _iter_case_dirs(root_dir):
    return sorted(
        path for path in root_dir.iterdir()
        if path.is_dir() and path.name.startswith(CASE_PREFIX)
    )


def collect_case_dirs(source_dir):
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    direct_cases = _iter_case_dirs(source_dir)
    if direct_cases:
        return direct_cases

    nested_cases = []
    for split_name in DEFAULT_SPLITS:
        split_dir = source_dir / split_name
        if split_dir.is_dir():
            nested_cases.extend(_iter_case_dirs(split_dir))

    if not nested_cases:
        raise FileNotFoundError(f"No BraTS case directories found under {source_dir}")

    unique_cases = {}
    for case_dir in nested_cases:
        unique_cases[case_dir.name] = case_dir
    return [unique_cases[case_id] for case_id in sorted(unique_cases)]


def split_case_dirs(case_dirs, train_ratio, val_ratio, test_ratio, seed):
    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = list(case_dirs)
    random.Random(int(seed)).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * float(train_ratio))
    val_count = int(total * float(val_ratio))
    test_count = total - train_count - val_count

    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError("Split ratios produced an empty split. Adjust the ratios or use more cases.")

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:train_count + val_count],
        "test": shuffled[train_count + val_count:],
    }


def ensure_clean_dir(path, clean):
    path = Path(path)
    if path.exists():
        if not clean:
            raise FileExistsError(
                f"Destination already exists: {path}. Use --clean to recreate it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_case(source_case_dir, dest_case_dir):
    source_case_dir = Path(source_case_dir).resolve()
    dest_case_dir = Path(dest_case_dir)
    if dest_case_dir.exists():
        return "exists"

    try:
        dest_case_dir.symlink_to(source_case_dir, target_is_directory=True)
        return "symlink"
    except (OSError, NotImplementedError, PermissionError):
        shutil.copytree(source_case_dir, dest_case_dir)
        return "copy"


def materialize_split(output_root, split_mapping):
    output_root = Path(output_root)
    link_stats = {"symlink": 0, "copy": 0, "exists": 0}

    for split_name, case_dirs in split_mapping.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for case_dir in case_dirs:
            mode = link_or_copy_case(case_dir, split_dir / case_dir.name)
            link_stats[mode] += 1

    return link_stats


def build_dev_subset(full_split_mapping, dev_size, seed):
    dev_size = int(dev_size)
    if dev_size <= 0:
        return None

    train_cases = list(full_split_mapping["train"])
    val_cases = list(full_split_mapping["val"])
    if dev_size > len(train_cases):
        raise ValueError(
            f"--dev_size={dev_size} exceeds available train cases ({len(train_cases)})."
        )

    rng = random.Random(int(seed) + 1)
    dev_train = sorted(rng.sample(train_cases, dev_size), key=lambda path: path.name)

    val_ratio = len(val_cases) / max(len(train_cases), 1)
    dev_val_size = max(1, int(round(dev_size * val_ratio)))
    dev_val_size = min(dev_val_size, len(val_cases))
    dev_val = sorted(rng.sample(val_cases, dev_val_size), key=lambda path: path.name)

    return {"train": dev_train, "val": dev_val}


def write_manifest(output_root, source_dir, split_mapping, seed, extra_meta=None):
    output_root = Path(output_root)
    manifest = {
        "source_dir": str(Path(source_dir).resolve()),
        "output_dir": str(output_root.resolve()),
        "seed": int(seed),
        "splits": {
            split_name: {
                "count": len(case_dirs),
                "case_ids": [case_dir.name for case_dir in case_dirs],
            }
            for split_name, case_dirs in split_mapping.items()
        },
    }
    if extra_meta:
        manifest.update(extra_meta)

    manifest_path = output_root / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main():
    args = parse_args()
    case_dirs = collect_case_dirs(args.source_dir)
    split_mapping = split_case_dirs(
        case_dirs=case_dirs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    ensure_clean_dir(args.out_dir, args.clean)
    full_link_stats = materialize_split(args.out_dir, split_mapping)
    full_manifest_path = write_manifest(
        output_root=args.out_dir,
        source_dir=args.source_dir,
        split_mapping=split_mapping,
        seed=args.seed,
        extra_meta={
            "mode": "full_split",
            "ratios": {
                "train": float(args.train_ratio),
                "val": float(args.val_ratio),
                "test": float(args.test_ratio),
            },
            "link_stats": full_link_stats,
        },
    )

    summary = {
        "source_dir": str(Path(args.source_dir).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "full_split": {
            split_name: len(case_dirs)
            for split_name, case_dirs in split_mapping.items()
        },
        "full_manifest": str(full_manifest_path.resolve()),
        "full_link_stats": full_link_stats,
    }

    dev_mapping = build_dev_subset(split_mapping, args.dev_size, args.seed)
    if dev_mapping is not None:
        ensure_clean_dir(args.dev_out_dir, args.clean)
        dev_link_stats = materialize_split(args.dev_out_dir, dev_mapping)
        dev_manifest_path = write_manifest(
            output_root=args.dev_out_dir,
            source_dir=args.out_dir,
            split_mapping=dev_mapping,
            seed=args.seed,
            extra_meta={
                "mode": "dev_subset",
                "derived_from": str(Path(args.out_dir).resolve()),
                "dev_size": int(args.dev_size),
                "link_stats": dev_link_stats,
            },
        )
        summary["dev_out_dir"] = str(Path(args.dev_out_dir).resolve())
        summary["dev_split"] = {
            split_name: len(case_dirs)
            for split_name, case_dirs in dev_mapping.items()
        }
        summary["dev_manifest"] = str(dev_manifest_path.resolve())
        summary["dev_link_stats"] = dev_link_stats

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
