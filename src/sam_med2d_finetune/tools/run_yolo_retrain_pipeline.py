import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from sam_med2d_finetune.tools.evaluate_yolo_recall import DEFAULT_CONF_VALUES
from sam_med2d_finetune.tools.train_yolo import sha256_file, write_json


CONF_VALUES = DEFAULT_CONF_VALUES
REMOTE_COMPLETE_MARKER = "REMOTE_PIPELINE_COMPLETE"
READY_MARKER = "READY_TO_POWER_OFF"
FAILED_MARKER = "PIPELINE_FAILED"
SHUTDOWN_REQUESTED_MARKER = "SHUTDOWN_REQUESTED"
SHUTDOWN_FAILED_MARKER = "SHUTDOWN_FAILED"


@dataclass(frozen=True)
class YoloConfig:
    tag: str
    imgsz: int
    batch: int
    mosaic: float
    scale: float
    box: float
    fraction: float = 1.0


SCREEN_CONFIGS = (
    YoloConfig("s1_img512_mosaic1_scale0p5_box7p5", 512, 32, 1.0, 0.5, 7.5, 1.0 / 3.0),
    YoloConfig("s2_img512_mosaic0_scale0p2_box7p5", 512, 32, 0.0, 0.2, 7.5, 1.0 / 3.0),
    YoloConfig("s3_img512_mosaic0_scale0p2_box10", 512, 32, 0.0, 0.2, 10.0, 1.0 / 3.0),
)


class PipelineError(RuntimeError):
    pass


class FormalY3GateFailure(PipelineError):
    pass


class PipelineTermination(BaseException):
    def __init__(self, signum):
        self.signum = int(signum)
        super().__init__(f"Received termination signal {self.signum}.")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the YOLO11m recall-oriented retraining pipeline with automatic Y3 selection."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run screen training, formal training and Y3 scans.")
    run_parser.add_argument("--data", required=True, help="Path to frozen YOLO data.yaml or dataset directory.")
    run_parser.add_argument("--model", required=True, help="Path to the YOLO11m base checkpoint.")
    run_parser.add_argument("--project", default="/root/autodl-tmp/runs")
    run_parser.add_argument(
        "--pipeline_dir",
        default="/root/autodl-tmp/runs/yolo11m_retrain_pipeline_seed11171",
    )
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--device", default="0")
    run_parser.add_argument("--ultralytics_dir", default="/root/autodl-tmp/.ultralytics")
    run_parser.add_argument("--seed", type=int, default=11171)
    run_parser.add_argument("--workers", type=int, default=2)
    run_parser.add_argument("--amp", default="true")
    run_parser.add_argument("--skip_amp_check", default="true")
    run_parser.add_argument("--screen_epochs", type=int, default=15)
    run_parser.add_argument("--formal_epochs", type=int, default=100)
    run_parser.add_argument("--patience", type=int, default=20)
    run_parser.add_argument("--save_period", type=int, default=10)
    run_parser.add_argument("--poll_seconds", type=float, default=30.0)
    run_parser.add_argument("--deadline_hours", type=float, default=30.0)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument(
        "--shutdown_on_exit",
        action="store_true",
        help="Call the AutoDL shutdown command after any terminal pipeline state is persisted.",
    )
    run_parser.add_argument(
        "--shutdown_command",
        default="/usr/bin/shutdown",
        help="AutoDL shutdown command. Do not pass probe arguments such as --help.",
    )
    run_parser.add_argument(
        "--shutdown_grace_seconds",
        type=float,
        default=10.0,
        help="Delay after flushing files and before replacing the process with the shutdown command.",
    )

    status_parser = subparsers.add_parser("status", help="Print manifest status and stale/process checks.")
    status_parser.add_argument("--pipeline_dir", required=True)
    status_parser.add_argument("--stale_minutes", type=float, default=30.0)

    ready_parser = subparsers.add_parser(
        "mark-ready",
        help="Create READY_TO_POWER_OFF after local reports are synced and verified.",
    )
    ready_parser.add_argument("--pipeline_dir", required=True)
    ready_parser.add_argument(
        "--require_sha",
        action="append",
        default=[],
        metavar="PATH=SHA256",
        help="Require a file to exist and match the supplied SHA-256 before writing READY_TO_POWER_OFF.",
    )
    return parser.parse_args()


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def marker_path(pipeline_dir, marker_name):
    return Path(pipeline_dir) / marker_name


def write_marker(pipeline_dir, marker_name, payload):
    path = marker_path(pipeline_dir, marker_name)
    path.write_text(json.dumps({"written_at": utc_now(), **payload}, indent=2), encoding="utf-8")
    return path


def write_ready_marker(pipeline_dir, required_sha):
    pipeline_dir = Path(pipeline_dir)
    if not marker_path(pipeline_dir, REMOTE_COMPLETE_MARKER).is_file():
        raise PipelineError(f"{READY_MARKER} requires {REMOTE_COMPLETE_MARKER}.")

    verified = []
    for item in required_sha:
        path_text, separator, expected_sha = item.partition("=")
        if not separator or not path_text or not expected_sha:
            raise ValueError(f"--require_sha must use PATH=SHA256, got: {item}")
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"Required verified file is missing: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise PipelineError(f"SHA mismatch for {path}: {actual_sha} != {expected_sha}")
        verified.append({"path": str(path.resolve()), "sha256": actual_sha})

    return write_marker(pipeline_dir, READY_MARKER, {"verified_files": verified})


def install_termination_handlers():
    previous_handlers = {}

    def handle_signal(signum, frame):
        del frame
        raise PipelineTermination(signum)

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


def request_shutdown(args, exit_code, reason):
    pipeline_dir = Path(args.pipeline_dir)
    payload = {
        "reason": reason,
        "exit_code": int(exit_code),
        "shutdown_command": str(args.shutdown_command),
    }
    update_pipeline_manifest(
        pipeline_dir,
        shutdown={
            "requested": True,
            **payload,
        },
    )
    write_marker(pipeline_dir, SHUTDOWN_REQUESTED_MARKER, payload)
    sys.stdout.flush()
    sys.stderr.flush()
    if hasattr(os, "sync"):
        os.sync()
    time.sleep(float(args.shutdown_grace_seconds))
    try:
        os.execv(str(args.shutdown_command), [str(args.shutdown_command)])
    except OSError as error:
        failure_payload = {
            **payload,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        update_pipeline_manifest(
            pipeline_dir,
            shutdown={
                "requested": True,
                "failed": True,
                **failure_payload,
            },
        )
        write_marker(pipeline_dir, SHUTDOWN_FAILED_MARKER, failure_payload)
        return False
    return True


def maybe_request_shutdown(args, exit_code, reason):
    if getattr(args, "command", None) != "run" or not getattr(args, "shutdown_on_exit", False):
        return False
    return request_shutdown(args, exit_code, reason)


def y3_gate_passed(result):
    return (
        int(result.get("fully_missed_case_count", -1)) == 0
        and float(result.get("slice_coverage_recall_0.50", 0.0)) >= 0.98
        and int(result.get("max_consecutive_missed_positive_slices", 10**9)) <= 2
    )


def choose_best_y3_result(summaries):
    candidates = []
    for checkpoint_name, summary in summaries.items():
        for result in summary.get("results", []):
            candidates.append({"checkpoint": checkpoint_name, **result})
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            item["fully_missed_case_count"],
            item["missed_positive_slice_count_coverage_0.50"],
            item["max_consecutive_missed_positive_slices"],
            -item["slice_coverage_recall_0.50"],
            item["background_false_positive_rate"],
            -item["mean_predicted_gt_box_area_ratio"],
            item["checkpoint"],
            item["iou"],
            item["conf"],
        ),
    )[0]


def y3_passed(summaries):
    best = choose_best_y3_result(summaries)
    return best is not None and y3_gate_passed(best), best


def read_epoch(results_csv):
    path = Path(results_csv)
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    epoch_value = rows[-1].get("epoch") or rows[-1].get("                  epoch")
    try:
        return int(float(epoch_value)) if epoch_value is not None else len(rows)
    except ValueError:
        return len(rows)


def collect_gpu_state():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def pid_is_running(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def is_manifest_stale(manifest, stale_minutes=30.0):
    updated_at = manifest.get("updated_at")
    if not updated_at:
        return True
    timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
    return age_seconds > stale_minutes * 60.0


def update_pipeline_manifest(pipeline_dir, **updates):
    pipeline_dir = Path(pipeline_dir)
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pipeline_dir / "pipeline_manifest.json"
    manifest = read_json(manifest_path, default={})
    manifest.update(updates)
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    return manifest


def base_train_args(args, config, run_name, epochs, fraction):
    command = [
        args.python,
        "-m",
        "sam_med2d_finetune.tools.train_yolo",
        "--data",
        args.data,
        "--model",
        args.model,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(config.imgsz),
        "--batch",
        str(config.batch),
        "--nbs",
        "64",
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--patience",
        str(args.patience),
        "--optimizer",
        "SGD",
        "--lr0",
        "0.01",
        "--momentum",
        "0.9",
        "--warmup_bias_lr",
        "0.0",
        "--project",
        args.project,
        "--name",
        run_name,
        "--purpose",
        "YOLO11m recall-oriented retraining pipeline",
        "--seed",
        str(args.seed),
        "--deterministic",
        "true",
        "--cache",
        "false",
        "--amp",
        args.amp,
        "--skip_amp_check",
        args.skip_amp_check,
        "--plots",
        "false",
        "--save_period",
        str(args.save_period),
        "--fraction",
        str(fraction),
        "--val",
        "true",
        "--mosaic",
        str(config.mosaic),
        "--scale",
        str(config.scale),
        "--box",
        str(config.box),
        "--hsv_h",
        "0.0",
        "--hsv_s",
        "0.0",
        "--hsv_v",
        "0.1",
        "--ultralytics_dir",
        args.ultralytics_dir,
    ]
    if args.resume:
        command.extend(["--resume", "true"])
    return command


def eval_args(args, checkpoint, out_dir, imgsz):
    return [
        args.python,
        "-m",
        "sam_med2d_finetune.tools.evaluate_yolo_recall",
        "--model",
        str(checkpoint),
        "--data",
        args.data,
        "--split",
        "val",
        "--conf_values",
        CONF_VALUES,
        "--iou",
        "0.60",
        "--imgsz",
        str(imgsz),
        "--device",
        args.device,
        "--max_det",
        "1",
        "--batch",
        "64",
        "--ultralytics_dir",
        args.ultralytics_dir,
        "--out_dir",
        str(out_dir),
    ]


def scan_log_for_terminal_failure(log_path):
    path = Path(log_path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if "outofmemoryerror" in text or "cuda out of memory" in text:
        return "oom"
    if " nan" in text or "nan," in text or "loss=nan" in text:
        return "nan"
    return None


def run_subprocess(command, log_path, pipeline_dir, stage, run_dir, poll_seconds, deadline_at):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{utc_now()}] stage={stage}\n")
        log_handle.write(" ".join(str(part) for part in command) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        update_pipeline_manifest(
            pipeline_dir,
            current_stage=stage,
            pid=process.pid,
            command=[str(part) for part in command],
            gpu_state=collect_gpu_state(),
        )
        while process.poll() is None:
            if time.time() > deadline_at:
                process.terminate()
                raise PipelineError(f"Deadline exceeded during {stage}.")
            epoch = read_epoch(Path(run_dir) / "results.csv")
            update_pipeline_manifest(
                pipeline_dir,
                current_stage=stage,
                pid=process.pid,
                epoch=epoch,
                gpu_state=collect_gpu_state(),
            )
            time.sleep(float(poll_seconds))
        exit_code = process.returncode

    terminal_failure = scan_log_for_terminal_failure(log_path)
    if exit_code != 0:
        raise PipelineError(f"{stage} failed with exit code {exit_code}.")
    if terminal_failure:
        raise PipelineError(f"{stage} produced terminal failure marker: {terminal_failure}.")


def read_y3_summaries(out_root):
    summaries = {}
    for checkpoint_name in ("best", "last"):
        summary_path = Path(out_root) / checkpoint_name / f"{checkpoint_name}_val" / "scan_summary.json"
        if summary_path.is_file():
            summaries[checkpoint_name] = json.loads(summary_path.read_text(encoding="utf-8"))
    return summaries


def run_y3(args, run_dir, out_root, imgsz, pipeline_dir, stage, deadline_at):
    out_root = Path(out_root)
    for checkpoint_name in ("best", "last"):
        checkpoint = Path(run_dir) / "weights" / f"{checkpoint_name}.pt"
        if not checkpoint.is_file():
            raise PipelineError(f"Missing checkpoint for Y3: {checkpoint}")
        run_subprocess(
            eval_args(args, checkpoint, out_root / checkpoint_name, imgsz),
            Path(pipeline_dir) / "logs" / f"{stage}_{checkpoint_name}.log",
            pipeline_dir,
            f"{stage}_{checkpoint_name}",
            run_dir,
            args.poll_seconds,
            deadline_at,
        )
    summaries = read_y3_summaries(out_root)
    if set(summaries) != {"best", "last"}:
        raise PipelineError(f"Y3 did not produce both best and last summaries under {out_root}.")
    return summaries


def build_640_config(source_config):
    return YoloConfig(
        f"{source_config.tag}_img640",
        640,
        16,
        source_config.mosaic,
        source_config.scale,
        source_config.box,
        1.0 / 3.0,
    )


def formal_config_from(screen_config):
    return YoloConfig(
        f"formal_{screen_config.tag}",
        screen_config.imgsz,
        screen_config.batch,
        screen_config.mosaic,
        screen_config.scale,
        screen_config.box,
        1.0,
    )


def artifact_hashes_for_run(run_dir):
    run_dir = Path(run_dir)
    hashes = {}
    for relative in (
        "manifest.json",
        "weights/best.pt",
        "weights/last.pt",
        "y3_detector_selection/best/best_val/scan_summary.json",
        "y3_detector_selection/last/last_val/scan_summary.json",
    ):
        path = run_dir / relative
        if path.is_file():
            hashes[relative] = sha256_file(path)
    return hashes


def run_pipeline(args):
    pipeline_dir = Path(args.pipeline_dir)
    deadline_at = time.time() + float(args.deadline_hours) * 3600.0
    update_pipeline_manifest(
        pipeline_dir,
        schema_version=1,
        pipeline="yolo11m_retrain_auto_y3",
        current_stage="starting",
        started_at=utc_now(),
        config={
            "screen_configs": [asdict(config) for config in SCREEN_CONFIGS],
            "conf_values": CONF_VALUES,
            "formal_epochs": args.formal_epochs,
            "screen_epochs": args.screen_epochs,
            "patience": args.patience,
        },
    )

    selected_screen = None
    screen_results = []
    configs_to_try = list(SCREEN_CONFIGS)
    fallback_640_tried = False

    while configs_to_try:
        config = configs_to_try.pop(0)
        run_name = f"screen_{config.tag}_seed{args.seed}"
        run_dir = Path(args.project) / run_name
        y3_out = run_dir / "y3_detector_selection"
        update_pipeline_manifest(
            pipeline_dir,
            current_stage=f"screen_train_{config.tag}",
            active_config=asdict(config),
        )
        run_subprocess(
            base_train_args(args, config, run_name, args.screen_epochs, config.fraction),
            pipeline_dir / "logs" / f"{run_name}.log",
            pipeline_dir,
            f"screen_train_{config.tag}",
            run_dir,
            args.poll_seconds,
            deadline_at,
        )
        summaries = run_y3(args, run_dir, y3_out, config.imgsz, pipeline_dir, f"screen_y3_{config.tag}", deadline_at)
        passed, best = y3_passed(summaries)
        screen_results.append({
            "config": asdict(config),
            "run_name": run_name,
            "run_dir": str(run_dir),
            "passed": passed,
            "best_y3": best,
        })
        update_pipeline_manifest(pipeline_dir, screen_results=screen_results)
        if passed and config.imgsz == 512:
            selected_screen = config
            break
        if not configs_to_try and selected_screen is None and not fallback_640_tried:
            if any(item["best_y3"] is None for item in screen_results):
                raise PipelineError("Cannot choose 640 fallback because a screen produced no Y3 results.")
            best_screen = min(
                screen_results,
                key=lambda item: (
                    item["best_y3"]["fully_missed_case_count"],
                    item["best_y3"]["missed_positive_slice_count_coverage_0.50"],
                    item["best_y3"]["max_consecutive_missed_positive_slices"],
                    -item["best_y3"]["slice_coverage_recall_0.50"],
                ),
            )
            configs_to_try.append(build_640_config(YoloConfig(**best_screen["config"])))
            fallback_640_tried = True
        elif passed:
            selected_screen = config
            break

    if selected_screen is None:
        raise PipelineError("No screen configuration passed Y3, including the 640 fallback.")

    formal_config = formal_config_from(selected_screen)
    formal_run_name = f"formal_{selected_screen.tag}_seed{args.seed}"
    formal_run_dir = Path(args.project) / formal_run_name
    update_pipeline_manifest(
        pipeline_dir,
        current_stage="formal_train",
        selected_config=asdict(formal_config),
        formal_run_name=formal_run_name,
    )
    run_subprocess(
        base_train_args(args, formal_config, formal_run_name, args.formal_epochs, 1.0),
        pipeline_dir / "logs" / f"{formal_run_name}.log",
        pipeline_dir,
        "formal_train",
        formal_run_dir,
        args.poll_seconds,
        deadline_at,
    )
    formal_summaries = run_y3(
        args,
        formal_run_dir,
        formal_run_dir / "y3_detector_selection",
        formal_config.imgsz,
        pipeline_dir,
        "formal_y3",
        deadline_at,
    )
    formal_passed, formal_best = y3_passed(formal_summaries)
    status = "succeeded" if formal_passed else "formal_y3_failed"
    artifact_hashes = artifact_hashes_for_run(formal_run_dir)
    update_pipeline_manifest(
        pipeline_dir,
        current_stage=status,
        pid=None,
        formal_y3={
            "passed": formal_passed,
            "best": formal_best,
            "summaries": {
                checkpoint: str(
                    formal_run_dir
                    / "y3_detector_selection"
                    / checkpoint
                    / f"{checkpoint}_val"
                    / "scan_summary.json"
                )
                for checkpoint in formal_summaries
            },
        },
        artifact_hashes=artifact_hashes,
    )
    write_marker(
        pipeline_dir,
        REMOTE_COMPLETE_MARKER,
        {
            "status": status,
            "formal_run_dir": str(formal_run_dir),
            "artifact_hashes": artifact_hashes,
        },
    )
    if not formal_passed:
        raise FormalY3GateFailure("Formal training completed, but best/last Y3 failed the frozen gate.")
    return 0


def status_payload(pipeline_dir, stale_minutes):
    manifest_path = Path(pipeline_dir) / "pipeline_manifest.json"
    manifest = read_json(manifest_path, default={})
    pid = manifest.get("pid")
    return {
        "manifest": str(manifest_path),
        "exists": manifest_path.is_file(),
        "current_stage": manifest.get("current_stage"),
        "updated_at": manifest.get("updated_at"),
        "stale": is_manifest_stale(manifest, stale_minutes) if manifest else True,
        "pid": pid,
        "pid_running": pid_is_running(pid),
        "remote_complete": marker_path(pipeline_dir, REMOTE_COMPLETE_MARKER).is_file(),
        "ready_to_power_off": marker_path(pipeline_dir, READY_MARKER).is_file(),
        "failed": marker_path(pipeline_dir, FAILED_MARKER).is_file(),
        "gpu_state": collect_gpu_state(),
    }


def main():
    args = parse_args()
    previous_signal_handlers = install_termination_handlers() if args.command == "run" else {}
    try:
        if args.command == "run":
            exit_code = run_pipeline(args)
            maybe_request_shutdown(args, exit_code, "pipeline_succeeded")
            return exit_code
        if args.command == "status":
            print(json.dumps(status_payload(args.pipeline_dir, args.stale_minutes), indent=2))
            return 0
        if args.command == "mark-ready":
            marker = write_ready_marker(args.pipeline_dir, args.require_sha)
            print(json.dumps({"ready_marker": str(marker.resolve())}, indent=2))
            return 0
        raise ValueError(f"Unsupported command: {args.command}")
    except BaseException as error:
        if isinstance(error, FormalY3GateFailure):
            print(str(error), file=sys.stderr)
            maybe_request_shutdown(args, 2, "formal_y3_gate_failed")
            return 2
        if getattr(args, "command", None) == "run":
            pipeline_dir = Path(args.pipeline_dir)
            if isinstance(error, PipelineTermination):
                exit_code = 128 + error.signum
                reason = "pipeline_terminated"
            elif isinstance(error, KeyboardInterrupt):
                exit_code = 130
                reason = "pipeline_interrupted"
            else:
                exit_code = 1
                reason = "pipeline_failed"
            update_pipeline_manifest(
                pipeline_dir,
                current_stage="failed",
                pid=None,
                exit_code=exit_code,
                failure={"type": type(error).__name__, "message": str(error)},
            )
            write_marker(
                pipeline_dir,
                FAILED_MARKER,
                {
                    "exit_code": exit_code,
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
            if maybe_request_shutdown(args, exit_code, reason):
                return exit_code
        raise
    finally:
        restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
