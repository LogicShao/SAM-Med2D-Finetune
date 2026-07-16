import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
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
    parser.add_argument("--nbs", type=int, default=None, help="Nominal batch size used by Ultralytics scaling.")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--optimizer",
        choices=(
            "auto",
            "SGD",
            "MuSGD",
            "Adam",
            "Adamax",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
        ),
        default="auto",
    )
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--warmup_bias_lr", type=float, default=None)
    parser.add_argument("--project", default="workdir_yolo", help="Ultralytics project/output root.")
    parser.add_argument("--name", default="brats_yolov8", help="Run name under the project directory.")
    parser.add_argument(
        "--purpose",
        default="Train a BraTS YOLO detector.",
        help="Declared purpose stored in manifest.json.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=str_to_bool, default=True)
    parser.add_argument("--cache", type=str_to_bool, default=False)
    parser.add_argument("--exist_ok", type=str_to_bool, default=False)
    parser.add_argument("--resume", type=str_to_bool, default=False)
    parser.add_argument("--amp", type=str_to_bool, default=True)
    parser.add_argument(
        "--skip_amp_check",
        type=str_to_bool,
        default=False,
        help=(
            "Skip Ultralytics' auxiliary-model AMP compatibility check. "
            "Use only after a hardware-specific finite-loss AMP smoke gate."
        ),
    )
    parser.add_argument("--plots", type=str_to_bool, default=True)
    parser.add_argument("--save_period", type=int, default=-1)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--val", type=str_to_bool, default=True)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--box", type=float, default=None)
    parser.add_argument("--hsv_h", type=float, default=None)
    parser.add_argument("--hsv_s", type=float, default=None)
    parser.add_argument("--hsv_v", type=float, default=None)
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


class TrainingTermination(BaseException):
    def __init__(self, signum):
        self.signum = int(signum)
        super().__init__(f"Received termination signal {self.signum}.")


def install_termination_handlers():
    previous_handlers = {}

    def handle_signal(signum, frame):
        del frame
        raise TrainingTermination(signum)

    for signal_name in ("SIGINT", "SIGTERM"):
        current_signal = getattr(signal, signal_name, None)
        if current_signal is None:
            continue
        previous_handlers[current_signal] = signal.getsignal(current_signal)
        signal.signal(current_signal, handle_signal)
    return previous_handlers


def restore_signal_handlers(previous_handlers):
    for current_signal, previous_handler in previous_handlers.items():
        signal.signal(current_signal, previous_handler)


def patch_ultralytics_dataset_threadpool():
    import ultralytics.data.dataset as dataset_module

    dataset_module.ThreadPool = SequentialThreadPool
    dataset_module.NUM_THREADS = 1


def patch_ultralytics_amp_check():
    import ultralytics.engine.trainer as trainer_module

    def assume_amp_compatible(model):
        del model
        return True

    trainer_module.check_amp = assume_amp_compatible


def dump_run_config(save_dir, payload):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / "train_config.json"
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)
    return path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _run_git_command(*args):
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def collect_git_state():
    revision = _run_git_command("rev-parse", "HEAD")
    porcelain = _run_git_command("status", "--porcelain")
    dirty_paths = porcelain.splitlines() if porcelain is not None else None
    entrypoint_path = Path(__file__).resolve()
    return {
        "revision": revision,
        "dirty_worktree": bool(dirty_paths) if dirty_paths is not None else None,
        "dirty_paths": dirty_paths,
        "entrypoint": {
            "path": str(entrypoint_path),
            "sha256": sha256_file(entrypoint_path),
        },
    }


def collect_environment():
    try:
        packages = sorted(
            f"{distribution.metadata['Name']}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        )
    except Exception:
        packages = []

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        }

    try:
        ultralytics_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        ultralytics_version = None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": ultralytics_version,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "gpu": gpu,
        "packages": packages,
    }


def load_adjacent_dataset_manifest(data_yaml):
    manifest_path = Path(data_yaml).resolve().parent / "dataset_manifest.json"
    if not manifest_path.is_file():
        return None, None, None
    return (
        manifest_path,
        json.loads(manifest_path.read_text(encoding="utf-8")),
        sha256_file(manifest_path),
    )


def summarize_dataset_exports(dataset_manifest):
    if not dataset_manifest:
        return None
    return {
        split_name: {
            key: value
            for key, value in split_payload.items()
            if key != "cases"
        }
        for split_name, split_payload in dataset_manifest.get("exports", {}).items()
    }


def resolve_run_dir(project, name, allow_existing=False):
    if not name or Path(name).name != name:
        raise ValueError("--name must be a single immutable run ID without path separators.")
    run_dir = Path(project).resolve() / name
    if allow_existing and not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume missing run directory: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Run directory already contains artifacts: {run_dir}. "
            "Use a new run ID, or explicitly pass --resume true."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def checkpoint_hashes(save_dir):
    weights_dir = Path(save_dir) / "weights"
    hashes = {}
    for name in ("best.pt", "last.pt"):
        checkpoint_path = weights_dir / name
        if checkpoint_path.is_file():
            hashes[name] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256_file(checkpoint_path),
            }
    return hashes


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
    if not 0.0 < float(args.fraction) <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}.")
    if args.skip_amp_check and not args.amp:
        raise ValueError("--skip_amp_check true requires --amp true.")
    optimizer_overrides = {
        "lr0": args.lr0,
        "momentum": args.momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
    }
    if args.optimizer == "auto" and any(value is not None for value in optimizer_overrides.values()):
        raise ValueError("Explicit optimizer hyperparameters require --optimizer other than auto.")
    if args.lr0 is not None and args.lr0 <= 0:
        raise ValueError(f"--lr0 must be positive, got {args.lr0}.")
    if args.momentum is not None and not 0.0 <= args.momentum < 1.0:
        raise ValueError(f"--momentum must be in [0, 1), got {args.momentum}.")
    if args.warmup_bias_lr is not None and args.warmup_bias_lr < 0:
        raise ValueError(f"--warmup_bias_lr must be non-negative, got {args.warmup_bias_lr}.")
    if args.nbs is not None and args.nbs <= 0:
        raise ValueError(f"--nbs must be positive, got {args.nbs}.")
    if args.mosaic is not None and args.mosaic < 0:
        raise ValueError(f"--mosaic must be non-negative, got {args.mosaic}.")
    if args.scale is not None and args.scale < 0:
        raise ValueError(f"--scale must be non-negative, got {args.scale}.")
    if args.box is not None and args.box <= 0:
        raise ValueError(f"--box must be positive, got {args.box}.")
    for name, value in {
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
    }.items():
        if value is not None and value < 0:
            raise ValueError(f"--{name} must be non-negative, got {value}.")
    data_yaml = resolve_data_yaml(args.data)
    ultralytics_dir = configure_ultralytics_env(args.ultralytics_dir)
    font_path = ensure_ultralytics_font(ultralytics_dir)
    normalized_data_yaml = normalize_data_yaml(data_yaml, ultralytics_dir)
    run_dir = resolve_run_dir(
        args.project,
        args.name,
        allow_existing=bool(args.resume),
    )
    dataset_manifest_path, dataset_manifest, dataset_manifest_sha256 = load_adjacent_dataset_manifest(
        data_yaml
    )
    model_path = Path(args.model)
    model_sha256 = sha256_file(model_path) if model_path.is_file() else None
    resume_checkpoint_path = run_dir / "weights" / "last.pt"
    resume_from_checkpoint = bool(args.resume and resume_checkpoint_path.is_file())
    train_kwargs = {
        "data": str(normalized_data_yaml),
        "epochs": int(args.epochs),
        "imgsz": int(args.imgsz),
        "batch": int(args.batch),
        "device": args.device,
        "workers": int(args.workers),
        "patience": int(args.patience),
        "optimizer": args.optimizer,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "seed": int(args.seed),
        "deterministic": bool(args.deterministic),
        "cache": bool(args.cache),
        # The wrapper owns collision checks and pre-creates the run directory
        # so manifest.json exists before Ultralytics starts.
        "exist_ok": True,
        "resume": resume_from_checkpoint,
        "amp": bool(args.amp),
        "plots": bool(args.plots),
        "save_period": int(args.save_period),
        "fraction": float(args.fraction),
        "val": bool(args.val),
    }
    train_kwargs.update(
        {key: value for key, value in optimizer_overrides.items() if value is not None}
    )
    train_kwargs.update(
        {
            key: value
            for key, value in {
                "mosaic": args.mosaic,
                "scale": args.scale,
                "box": args.box,
                "hsv_h": args.hsv_h,
                "hsv_s": args.hsv_s,
                "hsv_v": args.hsv_v,
                "nbs": args.nbs,
            }.items()
            if value is not None
        }
    )
    manifest_path = run_dir / "manifest.json"
    previous_manifest = None
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without an existing manifest: {manifest_path}")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("run_id") != args.name:
            raise ValueError(
                f"Resume manifest run ID mismatch: {previous_manifest.get('run_id')} != {args.name}"
            )
    config_payload = {
        "data_yaml": str(data_yaml),
        "normalized_data_yaml": str(normalized_data_yaml),
        "model": args.model,
        "ultralytics_dir": str(ultralytics_dir),
        "font_path": str(font_path) if font_path else None,
        "requested_exist_ok": bool(args.exist_ok),
        "skip_amp_check": bool(args.skip_amp_check),
        "train_kwargs": train_kwargs,
    }
    config_path = dump_run_config(run_dir, config_payload)
    started_at = utc_now()
    status_history = list(previous_manifest.get("status_history", [])) if previous_manifest else []
    if resume_from_checkpoint:
        start_event = "resume_checkpoint"
    elif previous_manifest:
        start_event = "retry_no_checkpoint"
    else:
        start_event = "start"
    status_history.append({
        "at": started_at,
        "status": "running",
        "event": start_event,
        "previous_status": previous_manifest.get("status") if previous_manifest else None,
    })
    manifest = {
        "schema_version": 1,
        "run_id": args.name,
        "stage": "yolo_detector_training",
        "method": "yolo",
        "purpose": args.purpose,
        "status": "running",
        "started_at": previous_manifest.get("started_at", started_at) if previous_manifest else started_at,
        "attempt_started_at": started_at,
        "status_history": status_history,
        "code": collect_git_state(),
        "dataset": {
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": sha256_file(data_yaml),
            "dataset_manifest": str(dataset_manifest_path) if dataset_manifest_path else None,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_seed": dataset_manifest.get("seed") if dataset_manifest else None,
            "splits": dataset_manifest.get("splits") if dataset_manifest else None,
            "export_summary": summarize_dataset_exports(dataset_manifest),
        },
        "base_checkpoint": {
            "path": str(model_path.resolve()) if model_path.is_file() else args.model,
            "sha256": model_sha256,
        },
        "resume_checkpoint": {
            "path": str(resume_checkpoint_path.resolve()),
            "sha256": sha256_file(resume_checkpoint_path),
        }
        if resume_from_checkpoint
        else None,
        "configuration": config_payload,
        "environment": collect_environment(),
        "artifacts": {
            "run_dir": str(run_dir.resolve()),
            "train_config": str(config_path.resolve()),
            "weights_dir": str((run_dir / "weights").resolve()),
        },
    }
    write_json(manifest_path, manifest)

    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    previous_signal_handlers = install_termination_handlers()

    try:
        from ultralytics import YOLO

        patch_ultralytics_dataset_threadpool()
        if args.skip_amp_check:
            patch_ultralytics_amp_check()
            print(
                "Skipping Ultralytics auxiliary-model AMP check; "
                "the run must pass a finite-loss AMP smoke gate."
            )
        model = YOLO(str(resume_checkpoint_path) if resume_from_checkpoint else args.model)
        results = model.train(**train_kwargs)
        save_dir = Path(
            getattr(results, "save_dir", getattr(getattr(model, "trainer", None), "save_dir", run_dir))
        ).resolve()
        if save_dir != run_dir.resolve():
            raise RuntimeError(f"Ultralytics wrote to unexpected run directory: {save_dir} != {run_dir.resolve()}")

        hashes = checkpoint_hashes(save_dir)
        manifest.update({
            "status": "succeeded",
            "exit_status": 0,
            "ended_at": utc_now(),
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            "selected_checkpoint": hashes.get("best.pt"),
            "artifact_hashes": hashes,
        })
        manifest["status_history"].append({
            "at": manifest["ended_at"],
            "status": "succeeded",
            "event": "complete",
        })
        write_json(manifest_path, manifest)
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, TrainingTermination))
        if isinstance(error, TrainingTermination):
            exit_status = 128 + error.signum
        elif isinstance(error, KeyboardInterrupt):
            exit_status = 130
        else:
            exit_status = 1
        manifest.update({
            "status": "interrupted" if interrupted else "failed",
            "exit_status": exit_status,
            "ended_at": utc_now(),
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "artifact_hashes": checkpoint_hashes(run_dir),
        })
        manifest["status_history"].append({
            "at": manifest["ended_at"],
            "status": manifest["status"],
            "event": "exception",
        })
        write_json(manifest_path, manifest)
        raise
    finally:
        restore_signal_handlers(previous_signal_handlers)

    print(json.dumps({
        "save_dir": str(run_dir.resolve()),
        "weights_dir": str((run_dir / "weights").resolve()),
        "best_pt": str((run_dir / "weights" / "best.pt").resolve()),
        "last_pt": str((run_dir / "weights" / "last.pt").resolve()),
        "train_config": str(config_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
