from __future__ import annotations

import re
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


STAGE_DEFINITIONS = (
    {"key": "received", "label": "已接收病例"},
    {"key": "inference", "label": "自动分割中"},
    {"key": "postprocess", "label": "后处理中"},
    {"key": "visualization", "label": "3D 重建中"},
    {"key": "terminal", "label": "已完成 / 失败"},
)

STAGE_LABELS = {item["key"]: item["label"] for item in STAGE_DEFINITIONS}

_RUNS: dict[str, dict[str, Any]] = {}
_THREADS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "case"


def build_run_id(case_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:6]
    return f"run_{timestamp}_{_sanitize_token(case_id)[:32]}_{suffix}"


def _build_stage_items() -> list[dict[str, str]]:
    return [
        {"key": item["key"], "label": item["label"], "state": "pending", "message": "等待执行"}
        for item in STAGE_DEFINITIONS
    ]


def _require_run(run_id: str) -> dict[str, Any]:
    if run_id not in _RUNS:
        raise KeyError(run_id)
    return _RUNS[run_id]


def _find_stage(job: dict[str, Any], stage_key: str) -> dict[str, Any]:
    for stage in job["stages"]:
        if stage["key"] == stage_key:
            return stage
    raise KeyError(stage_key)


def _touch(job: dict[str, Any]) -> None:
    job["updated_at"] = _now_text()


def register_thread(run_id: str, thread: threading.Thread) -> None:
    with _LOCK:
        _THREADS[run_id] = thread


def create_job(
    *,
    run_id: str,
    case_id: str,
    case_dir: Path,
    source_type: str,
    result_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    stages = _build_stage_items()
    stages[0]["state"] = "success"
    stages[0]["message"] = "病例文件已接收，等待进入自动分割"
    now_text = _now_text()
    job = {
        "run_id": run_id,
        "case_id": case_id,
        "case_dir": str(case_dir),
        "source_type": source_type,
        "result_dir": str(result_dir),
        "result_id": None,
        "result_url": None,
        "log_path": str(log_path),
        "status": "running",
        "message": "病例文件已接收，等待进入自动分割",
        "current_stage": "received",
        "current_stage_label": STAGE_LABELS["received"],
        "failed_stage": None,
        "failed_stage_label": None,
        "started_at": now_text,
        "updated_at": now_text,
        "completed_at": None,
        "stages": stages,
    }
    with _LOCK:
        _RUNS[run_id] = job
    return deepcopy(job)


def start_stage(run_id: str, stage_key: str, message: str) -> dict[str, Any]:
    with _LOCK:
        job = _require_run(run_id)
        stage = _find_stage(job, stage_key)
        stage["state"] = "running"
        stage["message"] = message
        job["status"] = "running"
        job["message"] = message
        job["current_stage"] = stage_key
        job["current_stage_label"] = STAGE_LABELS[stage_key]
        _touch(job)
        return deepcopy(job)


def finish_stage(run_id: str, stage_key: str, message: str) -> dict[str, Any]:
    with _LOCK:
        job = _require_run(run_id)
        stage = _find_stage(job, stage_key)
        stage["state"] = "success"
        stage["message"] = message
        job["message"] = message
        _touch(job)
        return deepcopy(job)


def complete_job(run_id: str, *, message: str, result_id: str, result_dir: Path) -> dict[str, Any]:
    with _LOCK:
        job = _require_run(run_id)
        terminal = _find_stage(job, "terminal")
        terminal["state"] = "success"
        terminal["message"] = message
        job["status"] = "success"
        job["message"] = message
        job["current_stage"] = "terminal"
        job["current_stage_label"] = STAGE_LABELS["terminal"]
        job["result_id"] = result_id
        job["result_dir"] = str(result_dir)
        job["result_url"] = f"/results/{result_id}"
        job["completed_at"] = _now_text()
        _touch(job)
        return deepcopy(job)


def fail_job(run_id: str, *, stage_key: str, message: str) -> dict[str, Any]:
    with _LOCK:
        job = _require_run(run_id)
        failed_stage = _find_stage(job, stage_key)
        failed_stage["state"] = "failed"
        failed_stage["message"] = message
        terminal = _find_stage(job, "terminal")
        terminal["state"] = "failed"
        terminal["message"] = "任务执行失败"
        job["status"] = "failed"
        job["message"] = message
        job["current_stage"] = stage_key
        job["current_stage_label"] = STAGE_LABELS.get(stage_key, stage_key)
        job["failed_stage"] = stage_key
        job["failed_stage_label"] = STAGE_LABELS.get(stage_key, stage_key)
        job["completed_at"] = _now_text()
        _touch(job)
        return deepcopy(job)


def get_job(run_id: str) -> dict[str, Any]:
    with _LOCK:
        return deepcopy(_require_run(run_id))

