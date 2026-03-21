from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from brats_case import BraTSCase
from postprocess_3d import postprocess_brats_masks
from web_demo.config import (
    DEFAULT_INFERENCE_ARGS,
    DEFAULT_POSTPROCESS_ARGS,
    DEMO_RUNS_DIR,
    PIPELINE_SUMMARY_FILE,
    PYTHON_EXECUTABLE,
    PROJECT_ROOT,
    UPLOAD_STAGE_DIR,
    ensure_web_demo_dirs,
)
from web_demo.services.results import encode_result_id, find_viewer_file


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "case"


def _list_modality_matches(case_dir: Path, modality: str) -> list[Path]:
    return sorted(case_dir.glob(f"*_{modality}.nii")) + sorted(case_dir.glob(f"*_{modality}.nii.gz"))


def _has_required_modalities(case_dir: Path) -> bool:
    return all(len(_list_modality_matches(case_dir, modality)) == 1 for modality in ("t1", "t1ce", "t2", "flair"))


def _resolve_single_case_dir(candidate: Path) -> Path:
    candidate = Path(str(candidate).strip().strip('"')).expanduser().resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"病例目录不存在: {candidate}")
    if _has_required_modalities(candidate):
        return candidate

    child_dirs = [path for path in candidate.iterdir() if path.is_dir() and _has_required_modalities(path)]
    if len(child_dirs) == 1:
        return child_dirs[0]
    raise ValueError("未找到可直接运行的单病例目录，请提供包含 t1/t1ce/t2/flair 的病例目录。")


def _infer_case_id_from_names(names: list[str]) -> str:
    case_ids = []
    pattern = re.compile(r"(.+?)_(t1ce|t1|t2|flair|seg)\.nii(\.gz)?$", re.IGNORECASE)
    for name in names:
        match = pattern.match(Path(name).name)
        if match:
            case_ids.append(match.group(1))
    if case_ids:
        return max(set(case_ids), key=case_ids.count)
    if names:
        return Path(names[0]).stem
    return "uploaded_case"


def _validate_case_dir(case_dir: Path) -> None:
    missing = [modality for modality in ("t1", "t1ce", "t2", "flair") if len(_list_modality_matches(case_dir, modality)) != 1]
    if missing:
        raise ValueError(f"病例文件不完整，缺少模态: {', '.join(missing)}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _init_pipeline_summary(case_input: dict[str, Any], result_dir: Path) -> dict[str, Any]:
    return {
        "case_id": case_input["case_id"],
        "result_dir": str(result_dir.resolve()),
        "source": {
            "type": case_input["source_type"],
            "case_dir": str(case_input["case_dir"]),
            "staged": bool(case_input.get("staged", False)),
        },
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "steps": [
            {"name": "自动分割", "status": "pending", "detail": "等待执行", "started_at": None, "finished_at": None},
            {"name": "后处理", "status": "pending", "detail": "等待执行", "started_at": None, "finished_at": None},
            {"name": "3D 重建", "status": "pending", "detail": "等待执行", "started_at": None, "finished_at": None},
        ],
        "artifacts": {},
    }


def _update_step(summary: dict[str, Any], step_name: str, status: str, detail: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    for step in summary["steps"]:
        if step["name"] != step_name:
            continue
        if status == "running" and step["started_at"] is None:
            step["started_at"] = now
        if status in {"done", "failed"}:
            if step["started_at"] is None:
                step["started_at"] = now
            step["finished_at"] = now
        step["status"] = status
        step["detail"] = detail
        return


def _build_result_dir(case_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEMO_RUNS_DIR / f"{timestamp}_{_sanitize_name(case_id)}"


def _build_combined_label(class_volumes: dict[str, np.ndarray]) -> np.ndarray:
    combined = np.zeros_like(class_volumes["WT"], dtype=np.uint8)
    combined[class_volumes["WT"] > 0] = 2
    combined[class_volumes["TC"] > 0] = 1
    combined[class_volumes["ET"] > 0] = 4
    return combined


def _load_binary_mask(path: Path) -> np.ndarray:
    return (np.asarray(nib.load(str(path)).dataobj) > 0).astype(np.uint8)


def prepare_case_input(case_dir_text: str | None, uploaded_files: list[Any] | None) -> dict[str, Any]:
    ensure_web_demo_dirs()

    if case_dir_text and str(case_dir_text).strip():
        case_dir = _resolve_single_case_dir(Path(case_dir_text))
        return {
            "case_id": case_dir.name,
            "case_dir": case_dir,
            "source_type": "local_path",
            "staged": False,
        }

    uploaded_files = [upload for upload in (uploaded_files or []) if getattr(upload, "filename", "")]
    if not uploaded_files:
        raise ValueError("请提供病例目录路径，或上传单病例所需的 NIfTI 文件。")

    case_id = _sanitize_name(_infer_case_id_from_names([upload.filename for upload in uploaded_files]))
    stage_root = UPLOAD_STAGE_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{case_id}"
    case_dir = stage_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    saved_names = set()
    for upload in uploaded_files:
        filename = Path(upload.filename).name
        if not filename.endswith((".nii", ".nii.gz")):
            continue
        if filename in saved_names:
            continue
        saved_names.add(filename)
        with (case_dir / filename).open("wb") as target_file:
            shutil.copyfileobj(upload.file, target_file)
        try:
            upload.file.close()
        except Exception:
            pass

    _validate_case_dir(case_dir)
    return {
        "case_id": case_id,
        "case_dir": case_dir.resolve(),
        "source_type": "uploaded_files",
        "staged": True,
    }


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=False,
    )
    if completed.returncode != 0:
        error_message = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(error_message)
    return completed


def run_inference(case_input: dict[str, Any]) -> dict[str, Any]:
    result_dir = Path(case_input["result_dir"])
    args = DEFAULT_INFERENCE_ARGS
    command = [
        str(PYTHON_EXECUTABLE),
        str(PROJECT_ROOT / "infer_volume.py"),
        "--case_dir",
        str(case_input["case_dir"]),
        "--output_dir",
        str(result_dir),
        "--sam_checkpoint",
        str(args["sam_checkpoint"]),
        "--finetuned_checkpoint",
        str(args["finetuned_checkpoint"]),
        "--finetune_method",
        str(args["finetune_method"]),
        "--prompt_mode",
        str(args["prompt_mode"]),
        "--model_type",
        str(args["model_type"]),
        "--image_size",
        str(args["image_size"]),
        "--input_channels",
        str(args["input_channels"]),
        "--encoder_adapter",
        str(args["encoder_adapter"]).lower(),
        "--device",
        str(args["device"]),
        "--threshold",
        str(args["threshold"]),
        "--use_amp",
        str(args["use_amp"]).lower(),
        "--yolo_checkpoint",
        str(args["yolo_checkpoint"]),
        "--yolo_conf",
        str(args["yolo_conf"]),
        "--yolo_iou",
        str(args["yolo_iou"]),
        "--yolo_max_det",
        str(args["yolo_max_det"]),
        "--yolo_topk",
        str(args["yolo_topk"]),
        "--prompt_box_strategy",
        str(args["prompt_box_strategy"]),
        "--z_prompt_mode",
        str(args["z_prompt_mode"]),
        "--wt_continuity_enabled",
        str(args["wt_continuity_enabled"]).lower(),
        "--postprocess",
        "false",
    ]
    completed = _run_command(command, cwd=PROJECT_ROOT)
    return {
        "result_dir": result_dir,
        "stdout": completed.stdout.strip(),
    }


def run_postprocess(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    case_meta_path = result_dir / "case_meta.json"
    if not case_meta_path.is_file():
        raise FileNotFoundError(f"未找到 case_meta.json: {case_meta_path}")

    case_meta = json.loads(case_meta_path.read_text(encoding="utf-8"))
    brats_case = BraTSCase.from_dir(Path(case_meta["case_dir"]))

    class_volumes = {
        "ET": _load_binary_mask(result_dir / "ET.nii.gz"),
        "TC": _load_binary_mask(result_dir / "TC.nii.gz"),
        "WT": _load_binary_mask(result_dir / "WT.nii.gz"),
    }
    processed_volumes, postprocess_report = postprocess_brats_masks(
        class_volumes=class_volumes,
        closing_radius=DEFAULT_POSTPROCESS_ARGS["closing_radius"],
        opening_radius=DEFAULT_POSTPROCESS_ARGS["opening_radius"],
        wt_keep_largest=DEFAULT_POSTPROCESS_ARGS["wt_keep_largest"],
        keep_topk_tc=DEFAULT_POSTPROCESS_ARGS["keep_topk_tc"],
        keep_topk_et=DEFAULT_POSTPROCESS_ARGS["keep_topk_et"],
        z_smooth_iterations=DEFAULT_POSTPROCESS_ARGS["z_smooth_iterations"],
    )
    post_combined_label = _build_combined_label(processed_volumes)

    for class_name, filename in (("ET", "post_ET.nii.gz"), ("TC", "post_TC.nii.gz"), ("WT", "post_WT.nii.gz")):
        brats_case.save_nifti(processed_volumes[class_name].astype(np.uint8), result_dir / filename)
    brats_case.save_nifti(post_combined_label.astype(np.uint8), result_dir / "post_combined_label.nii.gz")

    postprocess_report = {
        "case_id": brats_case.case_id,
        "output_dir": str(result_dir.resolve()),
        **postprocess_report,
    }
    report_path = result_dir / "postprocess_report.json"
    _write_json(report_path, postprocess_report)

    case_meta["postprocess_config"] = {"enabled": True, **DEFAULT_POSTPROCESS_ARGS}
    case_meta["postprocess_report_path"] = str(report_path.resolve())
    _write_json(case_meta_path, case_meta)
    return {"report_path": report_path}


def run_visualization(result_dir: Path) -> dict[str, Any]:
    result_dir = Path(result_dir)
    command = [
        str(PYTHON_EXECUTABLE),
        str(PROJECT_ROOT / "visualize_case.py"),
        "--output_dir",
        str(result_dir),
        "--mask_name",
        "all",
    ]
    completed = _run_command(command, cwd=PROJECT_ROOT)
    viewer_file = find_viewer_file(result_dir)
    if viewer_file is None:
        raise FileNotFoundError("3D HTML 预览未生成成功。")
    return {
        "viewer_file": str(viewer_file.resolve()),
        "stdout": completed.stdout.strip(),
    }


def run_full_pipeline(case_input: dict[str, Any]) -> dict[str, Any]:
    ensure_web_demo_dirs()
    result_dir = _build_result_dir(case_input["case_id"])
    result_dir.mkdir(parents=True, exist_ok=True)
    case_input = {**case_input, "result_dir": result_dir}

    summary = _init_pipeline_summary(case_input, result_dir)
    summary_path = result_dir / PIPELINE_SUMMARY_FILE
    _write_json(summary_path, summary)

    try:
        _update_step(summary, "自动分割", "running", "正在调用 infer_volume.py")
        _write_json(summary_path, summary)
        run_inference(case_input)
        _update_step(summary, "自动分割", "done", "已生成 ET / TC / WT / combined_label")
        summary["artifacts"]["case_meta"] = str((result_dir / "case_meta.json").resolve())
        _write_json(summary_path, summary)

        _update_step(summary, "后处理", "running", "正在生成 post_* 结果")
        _write_json(summary_path, summary)
        postprocess_result = run_postprocess(result_dir)
        _update_step(summary, "后处理", "done", "已完成 3D 后处理与层级约束")
        summary["artifacts"]["postprocess_report"] = str(postprocess_result["report_path"].resolve())
        _write_json(summary_path, summary)

        _update_step(summary, "3D 重建", "running", "正在调用 visualize_case.py")
        _write_json(summary_path, summary)
        visualization_result = run_visualization(result_dir)
        _update_step(summary, "3D 重建", "done", "HTML 预览已生成")
        summary["artifacts"]["viewer"] = visualization_result["viewer_file"]
        summary["status"] = "completed"
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(summary_path, summary)
    except Exception as exc:
        running_step = next((step for step in summary["steps"] if step["status"] == "running"), None)
        if running_step:
            _update_step(summary, running_step["name"], "failed", str(exc))
        summary["status"] = "failed"
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(summary_path, summary)
        raise

    return {
        "case_id": case_input["case_id"],
        "result_dir": result_dir,
        "result_id": encode_result_id(result_dir),
    }
