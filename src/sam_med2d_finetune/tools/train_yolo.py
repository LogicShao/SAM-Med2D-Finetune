import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import yaml


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO detector on the BraTS YOLO dataset.")
    parser.add_argument(
        "--data",
        default="datasets/brats_yolo/data.yaml",
        help="Path to data.yaml, or a dataset directory containing data.yaml.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO checkpoint or model name. Examples: yolov8n.pt, yolov8s.pt, path/to/last.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default="workdir_yolo", help="Ultralytics project/output root.")
    parser.add_argument("--name", default="brats_yolov8", help="Run name under the project directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=str_to_bool, default=False)
    parser.add_argument("--exist_ok", type=str_to_bool, default=False)
    parser.add_argument("--resume", type=str_to_bool, default=False)
    parser.add_argument("--amp", type=str_to_bool, default=True)
    parser.add_argument("--plots", type=str_to_bool, default=True)
    parser.add_argument("--save_period", type=int, default=-1)
    parser.add_argument(
        "--ultralytics_dir",
        default=".ultralytics",
        help="Writable directory for Ultralytics settings/cache files.",
    )
    return parser.parse_args()


def resolve_data_yaml(data_arg):
    data_path = Path(data_arg)
    if data_path.is_dir():
        data_path = data_path / "data.yaml"
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    return data_path.resolve()


def configure_ultralytics_env(config_dir):
    config_dir = Path(config_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLOv8_DIR", str(config_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(config_dir))
    return config_dir


def ensure_ultralytics_font(config_dir):
    config_dir = Path(config_dir).resolve()
    target_font = config_dir / "Arial.ttf"
    if target_font.is_file():
        return target_font

    candidate_fonts = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/Arial.ttf"),
    ]
    for candidate in candidate_fonts:
        if candidate.is_file():
            shutil.copyfile(candidate, target_font)
            return target_font
    return None


class SequentialThreadPool:
    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap(self, func=None, iterable=None, chunksize=1):
        if func is None or iterable is None:
            raise ValueError("SequentialThreadPool.imap requires both func and iterable.")
        for item in iterable:
            yield func(item)


def patch_ultralytics_dataset_threadpool():
    import ultralytics.data.dataset as dataset_module

    dataset_module.ThreadPool = SequentialThreadPool
    dataset_module.NUM_THREADS = 1


def dump_run_config(save_dir, payload):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "train_config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def normalize_data_yaml(data_yaml, ultralytics_dir):
    data_yaml = Path(data_yaml).resolve()
    dataset_root = data_yaml.parent
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YOLO dataset yaml: {data_yaml}")

    normalized = dict(data)
    normalized["path"] = str(dataset_root)

    normalized_yaml = Path(ultralytics_dir).resolve() / f"{dataset_root.name}_resolved_data.yaml"
    normalized_yaml.write_text(
        yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return normalized_yaml


def main():
    args = parse_args()
    data_yaml = resolve_data_yaml(args.data)
    ultralytics_dir = configure_ultralytics_env(args.ultralytics_dir)
    font_path = ensure_ultralytics_font(ultralytics_dir)
    normalized_data_yaml = normalize_data_yaml(data_yaml, ultralytics_dir)

    from ultralytics import YOLO

    patch_ultralytics_dataset_threadpool()

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(normalized_data_yaml),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": args.device,
        "workers": int(args.workers),
        "patience": int(args.patience),
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "seed": int(args.seed),
        "cache": bool(args.cache),
        "exist_ok": bool(args.exist_ok),
        "resume": bool(args.resume),
        "amp": bool(args.amp),
        "plots": bool(args.plots),
        "save_period": int(args.save_period),
    }

    results = model.train(**train_kwargs)
    save_dir = Path(getattr(results, "save_dir", getattr(getattr(model, "trainer", None), "save_dir", ""))).resolve()
    config_payload = {
        "data_yaml": str(data_yaml),
        "normalized_data_yaml": str(normalized_data_yaml),
        "model": args.model,
        "ultralytics_dir": str(ultralytics_dir),
        "font_path": str(font_path) if font_path else None,
        "train_kwargs": train_kwargs,
    }
    config_path = dump_run_config(save_dir, config_payload)

    print(json.dumps({
        "save_dir": str(save_dir),
        "weights_dir": str((save_dir / "weights").resolve()),
        "best_pt": str((save_dir / "weights" / "best.pt").resolve()),
        "last_pt": str((save_dir / "weights" / "last.pt").resolve()),
        "train_config": str(config_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
